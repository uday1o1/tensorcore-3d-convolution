"""End to end test of the dispatcher on real segmentation networks.

bench_end_to_end.py answers what the windowed paradigm costs if you adopt it
everywhere: on a standard kernel size 3 nnU-Net it is 2.34x slower on 3.9x the
memory. The crossover map says that is the wrong way to adopt it.

This benchmark asks the question the map actually poses. A dispatcher has to
clear two bars to be worth anything:

  1. do no harm. On a kernel size 3 network, where the windowed path loses in
     every shape measured, the dispatcher must match cuDNN. If it regresses
     here it is unusable, because small kernels are the common case.
  2. win where the map says it should. On a large kernel network, the
     dispatcher must beat cuDNN.

Both are measured on the same architecture, changing only the convolution
kernel size, so the comparison isolates kernel size rather than confounding it
with a different model.

A dispatcher that wins on bar 2 but fails bar 1 is not deployable, and
reporting only bar 2 would hide that. Both are printed either way.

Requires nnU-Net (via the MedNeXt fork) for the architecture.

Usage:  python bench_dispatch_network.py [kernel_sizes...]   default: 3 5
"""
import copy
import sys
from pathlib import Path
import time

import torch
import torch.nn as nn

# Resolve paths from this file, not the working directory, so the script
# runs from anywhere in a fresh clone.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dispatch import convert

from nnunet_mednext.network_architecture.generic_UNet import Generic_UNet

torch.backends.cudnn.benchmark = True

BASE_FEATURES = 30   # nnU-Net planner's value for this task, not the 32 default
NUM_CLASSES = 3      # background, liver, cancer
PATCH = 128


def build(k):
    """nnU-Net 3D, with every spatial convolution at kernel size k.

    nnU-Net's Generic_UNet derives padding as `1 if kernel == 3 else 0`, so it
    only supports kernel size 3 as a same-padded convolution. Asked for kernel
    size 5 it builds unpadded convolutions, the volume loses 4 voxels per
    convolution, and the encoder collapses to a 1 cubed feature map partway
    down, at which point InstanceNorm raises. That is a limitation of the
    architecture code rather than of the convolution, so we restore same
    padding after construction, which is what any real large kernel network
    would use.
    """
    net = Generic_UNet(
        1, BASE_FEATURES, NUM_CLASSES, 5, 2, 2,
        nn.Conv3d, nn.InstanceNorm3d, {"eps": 1e-5, "affine": True},
        nn.Dropout3d, {"p": 0, "inplace": True},
        nn.LeakyReLU, {"negative_slope": 1e-2, "inplace": True},
        False, False, lambda x: x, None,
        [[2, 2, 2]] * 5, [[k, k, k]] * 6, False, True, True,
    ).cuda().eval()
    fixed = 0
    for m in net.modules():
        if isinstance(m, nn.Conv3d) and m.kernel_size[0] > 1:
            want = tuple(kk // 2 for kk in m.kernel_size)
            if tuple(m.padding) != want:
                m.padding = want
                fixed += 1
    if fixed:
        print(f"  restored same padding on {fixed} convolutions "
              f"(Generic_UNet only pads kernel size 3)")
    return net


def measure(model, x, reps=15, warmup=5):
    def infer():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            model(x)
    for _ in range(warmup):
        infer()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(reps):
        infer()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps, torch.cuda.max_memory_allocated() / 1e9


def run_one(k, patch):
    print(f"\n{'='*66}\nnnU-Net 3D, kernel size {k}, patch {patch}^3\n{'='*66}")
    ref = build(k)
    x = torch.randn(1, 1, patch, patch, patch, device="cuda")

    # Exactness first: a policy that computes different numbers is not a
    # faster convolution, it is a different network.
    base = None
    results = {}
    for policy in ("cudnn", "windowed", "rule", "autotune"):
        model = copy.deepcopy(ref)
        total, eligible = convert(model, policy=policy)
        with torch.no_grad():
            torch.backends.cudnn.allow_tf32 = False
            out = model(x)
            torch.backends.cudnn.allow_tf32 = True
        out = out[0] if isinstance(out, (list, tuple)) else out
        if base is None:
            base = out.detach().clone()
            rel = 0.0
        else:
            rel = (out - base).abs().max().item() / max(base.abs().max().item(), 1e-12)

        # The exactness check above runs in fp32 with TF32 off, and the
        # autotune policy decides on its first forward pass. Timing then runs
        # under fp16 autocast. Left alone, autotune picks a path for one
        # precision regime and is measured in another, and it showed up as
        # autotune scoring 0.425x, identical to always-windowed and worse than
        # the rule, which is impossible for a policy that is supposed to be an
        # upper bound. Clear the cache and let it decide under autocast.
        for m in model.modules():
            if hasattr(m, "_choice"):
                m._choice.clear()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            model(x)

        dt, mem = measure(model, x)
        chosen = sum(1 for m in model.modules()
                     if hasattr(m, "_choice") and any(m._choice.values()))
        results[policy] = (dt, mem, chosen, rel)
        print(f"  {policy:9s} {dt*1000:8.1f} ms   peak {mem:5.2f} GB   "
              f"windowed on {chosen:2d}/{eligible} eligible layers   rel_err {rel:.2e}")
        del model
        torch.cuda.empty_cache()

    c = results["cudnn"][0]
    print(f"\n  speedup over cuDNN:")
    for policy in ("windowed", "rule", "autotune"):
        print(f"    {policy:9s} {c/results[policy][0]:5.3f}x")
    del ref, x
    torch.cuda.empty_cache()
    return {p: results[p][0] for p in results}


def main(kernels):
    patch = PATCH
    out = {}
    for k in kernels:
        # large kernels at 128^3 exceed 24GB; shrink the patch and say so
        p = patch if k <= 3 else 64
        if p != patch:
            print(f"\n(note: kernel size {k} uses patch {p}^3, since {patch}^3 "
                  f"does not fit in 24GB at this kernel size)")
        try:
            out[k] = run_one(k, p)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  kernel size {k}: out of memory even at patch {p}")

    print(f"\n{'='*66}\nVERDICT\n{'='*66}")
    for k, r in out.items():
        rule_vs_cudnn = r["cudnn"] / r["rule"]
        if k <= 3:
            ok = rule_vs_cudnn > 0.97   # within 3 percent counts as no harm
            print(f"  k={k}: bar 1, do no harm. rule is {rule_vs_cudnn:.3f}x of "
                  f"cuDNN -> {'PASS' if ok else 'FAIL'}")
        else:
            ok = rule_vs_cudnn > 1.0
            print(f"  k={k}: bar 2, win at large kernels. rule is "
                  f"{rule_vs_cudnn:.3f}x of cuDNN -> {'PASS' if ok else 'FAIL'}")
        print(f"        always-windowed would have been "
              f"{r['cudnn']/r['windowed']:.3f}x, autotune {r['cudnn']/r['autotune']:.3f}x")
    return 0


if __name__ == "__main__":
    ks = [int(a) for a in sys.argv[1:]] or [3, 5]
    sys.exit(main(ks))
