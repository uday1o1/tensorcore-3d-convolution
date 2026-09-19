"""
Substitutes the windowed convolution into a whole nnU-Net and measures what
the paradigm costs in practice: inference latency, peak memory, and whether
the network output changes.

The substitution runs on identical weights, so any output difference is the
kernel's numerics alone, with no training variance to confound it.

Also profiles where GPU time actually goes, which bounds what any convolution
kernel can address. FLOP share badly overstates this: convolution is 99.6
percent of this network's FLOPs but 65.6 percent of its inference time.

A trap this script exists to avoid: an earlier version reported identical
timings for both configurations because `child.dilation` is the tuple
`(1,1,1)` and `(1,1,1) == 1` is False, which silently disabled the windowed
path on every layer and measured cuDNN against itself. The substitution count
is printed so that cannot happen unnoticed.

Usage:  python bench_end_to_end.py
"""
import copy
import sys
import time

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, "../src")
from winconv import WindowedConv3d

from nnunet_mednext.network_architecture.generic_UNet import Generic_UNet

torch.backends.cudnn.benchmark = True

BASE_FEATURES = 30   # nnU-Net planner's value for this task, not the 32 default
NUM_CLASSES = 3      # background, liver, cancer
PATCH = 128


def build():
    return Generic_UNet(
        1, BASE_FEATURES, NUM_CLASSES, 5, 2, 2,
        nn.Conv3d, nn.InstanceNorm3d, {"eps": 1e-5, "affine": True},
        nn.Dropout3d, {"p": 0, "inplace": True},
        nn.LeakyReLU, {"negative_slope": 1e-2, "inplace": True},
        False, False, lambda x: x, None,
        [[2, 2, 2]] * 5, [[3, 3, 3]] * 6, False, True, True,
    ).cuda().eval()


def substitute(model):
    """Replace every Conv3d with WindowedConv3d, preserving weights exactly."""
    total = windowed = 0
    for _, mod in model.named_modules():
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Conv3d):
                new = WindowedConv3d(
                    child.in_channels, child.out_channels, child.kernel_size,
                    child.stride, child.padding, child.dilation, child.groups,
                    child.bias is not None,
                ).cuda()
                new.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    new.bias.data = child.bias.data.clone()
                setattr(mod, name, new)
                total += 1
                windowed += int(new.use_windowed)
    return total, windowed


def profile_split(fn, label):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
    evts = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    total = sum(e.self_device_time_total for e in evts)
    buckets = {"conv": 0, "layout": 0, "norm": 0, "other": 0}
    for e in evts:
        k = e.key
        if "xmma" in k or "convNd" in k or "implicit_gemm" in k:
            buckets["conv"] += e.self_device_time_total
        elif any(s in k for s in ("ToNhwc", "ToNchw", "Memcpy", "Memset")):
            buckets["layout"] += e.self_device_time_total
        elif "bn_fw" in k or "bn_bw" in k:
            buckets["norm"] += e.self_device_time_total
        else:
            buckets["other"] += e.self_device_time_total
    parts = "  ".join(f"{n} {v/total*100:.1f}%" for n, v in buckets.items())
    print(f"  {label}: {parts}")


def main():
    torch.manual_seed(0)
    ref = build()
    win = copy.deepcopy(ref)
    total, windowed = substitute(win)
    print(f"substituted {total} Conv3d layers, {windowed} use the windowed path, "
          f"{total - windowed} fall back")
    if windowed == 0:
        print("ERROR: windowed path never active, this would measure cuDNN against itself")
        return 1

    x = torch.randn(1, 1, PATCH, PATCH, PATCH, device="cuda")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.no_grad():
        a, b = ref(x), win(x)
    rel = (a - b).abs().max().item() / a.abs().max().item()
    print(f"full network output match on identical weights: rel_err {rel:.2e}")
    torch.backends.cudnn.allow_tf32 = True

    def infer(m):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            m(x)

    print()
    times = {}
    for model, label in ((ref, "cuDNN   "), (win, "windowed")):
        for _ in range(5):
            infer(model)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(15):
            infer(model)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 15
        times[label.strip()] = dt
        print(f"{label}: {dt*1000:7.1f} ms/patch   peak {torch.cuda.max_memory_allocated()/1e9:5.2f} GB")

    print(f"\nend to end ratio (cuDNN/windowed): {times['cuDNN']/times['windowed']:.3f}x")
    print("\nGPU time breakdown, which bounds what any conv kernel can address:")
    profile_split(lambda: infer(ref), "cuDNN   ")
    profile_split(lambda: infer(win), "windowed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
