"""Where should a large 3D receptive field be spent?

Enlarging every convolution kernel in a volumetric network is not the only way
to buy receptive field, and it is close to the worst way to pay for it. Cost is
not uniform across stages: an early stage carries a large spatial extent, so
enlarging its kernel is expensive, while a deep stage has already been
downsampled and enlarging its kernel is nearly free. If receptive field bought
in a deep stage is worth anything like receptive field bought in an early one,
selective placement should dominate uniform enlargement.

This measures the cost side of that question for MedNeXt, whose large kernel
convolution is depthwise, using only standard PyTorch and cuDNN operations. The
accuracy side needs training and is produced by the companion runs.

COST PREDICTION, recorded before any accuracy measurement, derived from
cuDNN's depthwise times in results/crossover_map_rtx3090.json:

  Approximating a four level encoder as (30ch, 64), (60ch, 32), (120ch, 16),
  (240ch, 16), the measured depthwise cost of each placement relative to an
  all-kernel-3 network is:

      3-3-3-3   1.00x
      3-3-5-5   1.16x
      3-5-5-7   1.84x
      5-5-5-5   4.59x

  The first stage alone accounts for 79 percent of the all-kernel-3 depthwise
  cost, which is why enlarging it dominates the bill.

  We therefore predict that 3-3-5-5 lies on the accuracy-cost Pareto frontier
  and 5-5-5-5 does not. For uniform enlargement to be justified it would have
  to deliver roughly four times the Dice improvement of deep-stage-only
  placement, and MedNeXt reports kernel 5 with UpKern gaining about 0.2 Dice
  points over kernel 3 (MedNeXt-B: 84.01 to 84.23 on BTCV, 89.14 to 89.38 on
  AMOS22), not multiples.

  The prediction fails if Dice turns out to depend on receptive field in the
  EARLY stages specifically, where the cost is concentrated. That is the
  outcome that would make uniform enlargement worth its price, and it is the
  one this experiment is designed to detect.

Reports per configuration: parameters, FLOPs, measured latency, measured peak
training memory, and the theoretical receptive field in voxels.

Usage:  python bench_receptive_field.py [--patch 128] [--controls] [--tag NAME]

The committed results/receptive_field_cost.json keeps the placement, its
kernels, the cost fields, the retuned-layer count and the GPU name, and no
receptive field values; the receptive field values the write-up quotes come
from results/effective_receptive_field.json.
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from benchenv import results_dir, device_meta
from kernel_placement import set_stage_kernels, receptive_field

# The deep-stage placements under study, plus the uniform-5 upper bound they
# are compared against.
PLACEMENTS = {
    "3-3-3-3": [3, 3, 3, 3],
    "3-3-5-5": [3, 3, 5, 5],
    "3-5-5-7": [3, 5, 5, 7],
    "5-5-5-5": [5, 5, 5, 5],
}
# Early-placement controls: the deep placements mirrored, so the same kernel
# sizes sit in the shallow stages instead. The mirrors enlarge fewer layers
# (the bottleneck takes the deepest stage's kernel), so a deep placement that
# still costs less than its mirror shows that depth, not the number of
# enlarged layers, sets the price.
CONTROLS = {
    "5-5-3-3": [5, 5, 3, 3],
    "7-5-5-3": [7, 5, 5, 3],
}
PATCH = 64
BATCH = 2


def count_flops_params(model, x):
    """FLOPs for conv layers via shape bookkeeping, plus parameter count.

    Counted by hooks rather than estimated, so grouped and depthwise
    convolutions are charged per group: a depthwise layer reduces over one
    input channel, not over all of them, and a FLOP model that misses this
    overstates it by the channel count. Transposed convolutions are charged at
    output resolution, which overstates them by the upsampling factor.
    """
    flops = [0]

    def hook(mod, inp, out):
        if isinstance(mod, (nn.Conv3d, nn.ConvTranspose3d)):
            cin_per_group = mod.in_channels // mod.groups
            k = 1
            for d in mod.kernel_size:
                k *= d
            spatial = 1
            for d in out.shape[2:]:
                spatial *= d
            flops[0] += 2 * out.shape[1] * cin_per_group * k * spatial * out.shape[0]

    handles = [m.register_forward_hook(hook) for m in model.modules()]
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    params = sum(p.numel() for p in model.parameters())
    return flops[0], params


def measure(model, x, train=False, reps=10, warmup=3):
    def step():
        if train:
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
                out = out[0] if isinstance(out, (list, tuple)) else out
                loss = out.float().pow(2).mean()
            loss.backward()
        else:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                model(x)
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(reps):
        step()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps, torch.cuda.max_memory_allocated() / 1e9


def build(kernels):
    """MedNeXt-B with per-stage depthwise kernel sizes."""
    from nnunet_mednext.network_architecture.mednextv1.create_mednext_v1 import (
        create_mednextv1_base)
    model = create_mednextv1_base(num_input_channels=1, num_classes=3,
                                  kernel_size=3, ds=False).cuda()
    changed = set_stage_kernels(model, kernels)
    uniform = len(set(kernels)) == 1 and kernels[0] == 3
    if changed == 0 and not uniform:
        raise RuntimeError(f"no kernels changed for {kernels}, "
                           "every configuration would measure the same network")
    return model.eval(), changed


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--patch", type=int, default=PATCH,
                    help="cubic patch edge in voxels (default 64; training used 128)")
    ap.add_argument("--controls", action="store_true",
                    help="also measure the early-placement controls 5-5-3-3 and 7-5-5-3")
    ap.add_argument("--tag", default="",
                    help="output receptive_field_cost_<tag>.json instead of "
                         "receptive_field_cost.json")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs a CUDA device")
        return 1
    patch = args.patch
    placements = {**PLACEMENTS, **(CONTROLS if args.controls else {})}
    x = torch.randn(BATCH, 1, patch, patch, patch, device="cuda")
    rows = []
    print(f"MedNeXt-B, patch {patch}^3, batch {BATCH}, "
          f"{torch.cuda.get_device_name(0)}\n")
    print(f"{'placement':<12}{'params M':>10}{'GFLOPs':>10}{'infer ms':>10}"
          f"{'train ms':>10}{'train GB':>10}{'RF vox':>8}")
    for name, kernels in placements.items():
        try:
            model, changed = build(kernels)
            fl, pr = count_flops_params(model, x)
            inf, _ = measure(model, x, train=False)
            model.train()
            tr, mem = measure(model, x, train=True)
            rf, blocks = receptive_field(model, kernels)
            rows.append({"placement": name, "kernels": kernels,
                         "params_m": pr / 1e6, "gflops": fl / 1e9,
                         "infer_ms": inf * 1000, "train_ms": tr * 1000,
                         "train_gb": mem, "receptive_field_voxels": rf,
                         "layers_retuned": changed, "patch": patch,
                         "batch": BATCH, **device_meta()})
            print(f"{name:<12}{pr/1e6:>10.2f}{fl/1e9:>10.1f}{inf*1000:>10.1f}"
                  f"{tr*1000:>10.1f}{mem:>10.2f}{rf:>8d}")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            torch.cuda.empty_cache()
            print(f"{name:<12}  failed: {type(e).__name__}: {e}")

    if rows:
        out = f"receptive_field_cost{'_' + args.tag if args.tag else ''}.json"
        json.dump(rows, open(results_dir() / out, "w"), indent=1)
        base = next((r for r in rows if r["placement"] == "3-3-3-3"), rows[0])
        print(f"\nrelative to {base['placement']}:")
        for r in rows:
            print(f"  {r['placement']:<12} FLOPs {r['gflops']/base['gflops']:5.2f}x   "
                  f"train time {r['train_ms']/base['train_ms']:5.2f}x   "
                  f"memory {r['train_gb']/base['train_gb']:5.2f}x   "
                  f"receptive field {r['receptive_field_voxels']/base['receptive_field_voxels']:5.2f}x")
        print("\nThe accuracy axis comes from the training runs; this file is the"
              "\ncost axis they are plotted against.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
