"""
Correctness verification for every convolution implementation in src/.

This exists because the failure mode observed in the published kernels this
project reproduces is SILENT: they return zero or partially computed output
with no error raised. An early benchmark sweep here reported 13,483 TFLOPS on
one shape, roughly 190x the device peak, which is what timing an empty kernel
launch measures. No timing in this project is reported for a configuration
that has not first passed correctness.

Note on TF32: PyTorch enables TF32 on Ampere by default, and its 10-bit
mantissa produces relative errors near 3e-04 that look like algorithmic bugs
but are not. Verification must disable it or it will misattribute precision
to correctness.

Usage:  python verify_correctness.py
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "../src")
from winconv import (im2win_conv2d, im2win_conv3d, im2win_conv3d_minmat,
                     im2win_conv3d_depthwise, WindowedConv3d)
from fftconv import fft_conv3d

# true fp32 on both sides, otherwise TF32 masquerades as error
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

TOL = 1e-5


def check(label, got, ref):
    if got.shape != ref.shape:
        print(f"  FAIL {label}: shape {tuple(got.shape)} vs {tuple(ref.shape)}")
        return False
    rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-12)
    ok = rel < TOL
    print(f"  {'OK  ' if ok else 'FAIL'} {label}: rel_err {rel:.2e}")
    return ok


def main():
    torch.manual_seed(0)
    allok = True

    print("2D windowed convolution")
    for B, C, H, W, Co, k, s in [(2, 8, 16, 16, 4, 3, 1), (1, 3, 9, 9, 5, 3, 1),
                                 (4, 16, 32, 32, 32, 3, 1), (2, 30, 24, 24, 60, 3, 1),
                                 (2, 8, 16, 16, 4, 5, 1), (2, 8, 17, 19, 4, 3, 2)]:
        x = torch.randn(B, C, H, W, device="cuda")
        w = torch.randn(Co, C, k, k, device="cuda")
        allok &= check(f"C{C} {H}x{W} Co{Co} k{k} s{s}",
                       im2win_conv2d(x, w, stride=s), F.conv2d(x, w, stride=s))

    print("3D windowed convolution, k^2 materialization")
    for B, C, D, Co, k, s in [(2, 4, 8, 3, 3, 1), (2, 30, 16, 60, 3, 1), (2, 8, 10, 4, 3, 2)]:
        x = torch.randn(B, C, D, D, D, device="cuda")
        w = torch.randn(Co, C, k, k, k, device="cuda")
        allok &= check(f"C{C} {D}^3 Co{Co} k{k} s{s}",
                       im2win_conv3d(x, w, stride=s), F.conv3d(x, w, stride=s))

    print("3D windowed convolution, minimal (k) materialization")
    for B, C, D, Co, k, s in [(2, 4, 8, 3, 3, 1), (2, 30, 16, 60, 3, 1),
                              (2, 16, 20, 32, 5, 1), (1, 6, 20, 4, 7, 1), (2, 8, 10, 4, 3, 2)]:
        x = torch.randn(B, C, D, D, D, device="cuda")
        w = torch.randn(Co, C, k, k, k, device="cuda")
        allok &= check(f"C{C} {D}^3 Co{Co} k{k} s{s}",
                       im2win_conv3d_minmat(x, w, stride=s), F.conv3d(x, w, stride=s))

    print("3D windowed convolution, depthwise")
    # the layer type large-kernel 3D architectures actually use; the grouping
    # here is easy to get silently wrong, since it depends on the unfold
    # ordering the materialized channel axis as (c, kd) rather than (kd, c)
    for B, C, D, k, s in [(2, 8, 12, 3, 1), (2, 32, 16, 5, 1),
                          (1, 64, 20, 7, 1), (2, 16, 17, 3, 2)]:
        x = torch.randn(B, C, D, D, D, device="cuda")
        w = torch.randn(C, 1, k, k, k, device="cuda")
        allok &= check(f"C{C} {D}^3 k{k} s{s} groups={C}",
                       im2win_conv3d_depthwise(x, w, stride=s),
                       F.conv3d(x, w, stride=s, groups=C))

    print("FFT convolution")
    for B, C, D, Co, k in [(1, 4, 12, 3, 3), (2, 8, 16, 6, 5), (1, 6, 14, 4, 7)]:
        x = torch.randn(B, C, D, D, D, device="cuda")
        w = torch.randn(Co, C, k, k, k, device="cuda")
        allok &= check(f"C{C} {D}^3 Co{Co} k{k}", fft_conv3d(x, w), F.conv3d(x, w))

    print("WindowedConv3d drop-in module vs nn.Conv3d (identical weights)")
    import torch.nn as nn
    for ci, co, k, p, s, g in [(30, 60, 3, 1, 1, 1), (120, 120, 3, 1, 1, 1),
                               (60, 60, 3, 1, 2, 1), (32, 64, 1, 0, 1, 1),
                               (64, 64, 5, 2, 1, 64),   # depthwise, windowed
                               (64, 64, 3, 1, 1, 4)]:   # partial groups, falls back
        ref = nn.Conv3d(ci, co, k, stride=s, padding=p, groups=g).cuda()
        got = WindowedConv3d(ci, co, k, stride=s, padding=p, groups=g).cuda()
        got.weight.data = ref.weight.data.clone()
        got.bias.data = ref.bias.data.clone()
        x = torch.randn(2, ci, 16, 16, 16, device="cuda")
        allok &= check(f"conv3d({ci},{co},k={k},p={p},s={s},g={g}) windowed={got.use_windowed}",
                       got(x), ref(x))

    print("DispatchingConv3d vs nn.Conv3d, every policy (identical weights)")
    # Every policy must return the SAME numbers; they differ only in which
    # kernel computes them. A policy that silently never selects the windowed
    # path would pass a correctness check while measuring cuDNN against
    # itself, so the selected path is printed alongside the error.
    from dispatch import DispatchingConv3d, prefer_windowed
    for policy in ("cudnn", "windowed", "rule", "autotune"):
        for ci, co, k, p, s, g in [(240, 240, 5, 2, 1, 1), (60, 60, 3, 1, 1, 1),
                                   (64, 64, 5, 2, 1, 64)]:
            ref = nn.Conv3d(ci, co, k, stride=s, padding=p, groups=g).cuda()
            got = DispatchingConv3d(ci, co, k, stride=s, padding=p, groups=g,
                                    policy=policy).cuda()
            got.weight.data = ref.weight.data.clone()
            got.bias.data = ref.bias.data.clone()
            x = torch.randn(1, ci, 16, 16, 16, device="cuda")
            out = got(x)
            chose = list(got._choice.values())[0]
            allok &= check(f"{policy:8s} conv3d({ci},{co},k={k},g={g}) "
                           f"chose={'windowed' if chose else 'cudnn':8s}", out, ref(x))

    # The rule must actually discriminate, or it is not a rule.
    decisions = {prefer_windowed(C, sp, k)
                 for C in (30, 240) for sp in (16, 64) for k in (3, 11)}
    if len(decisions) < 2:
        print("  FAIL prefer_windowed returns a constant, it is not deciding anything")
        allok = False
    else:
        print("  OK   prefer_windowed discriminates across shapes")

    print()
    print("ALL CORRECT" if allok else "FAILURES PRESENT")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
