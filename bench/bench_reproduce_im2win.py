"""
Reproduces the published Im2win comparison with complete accounting.

Produces three numbers:
  1. the kernel-only ratio, measured the way the published comparison is
  2. the cost of the window materialization the published timing excludes
  3. the end-to-end ratio once that cost is included

The published harness times the convolution kernel alone. The image2window
transform that produces the kernel's input runs on the host before the timing
loop and is reused across all fifty timed iterations, while allocation and
transfer costs accumulate into a separate variable that is written to the log
but excluded from the reported TFLOPS. cuDNN's
implicit GEMM performs the equivalent rearrangement inside its timed kernel
and is charged for it in full.

The released transform is host-side OpenMP, so a usable integration has to
implement it on the device. That is what is measured here.

The kernel-only number is the constant IM2WIN_TFLOPS, measured on an RTX 3090
with the sweep harness built from bench/sweep_convs.cpp; this script does not run the Im2win kernel. To
re-measure it on another device, build the kernel in an Im2win clone (see
bench/CMakeLists.txt). The cuDNN and transform measurements are self-contained.

Usage:  python bench_reproduce_im2win.py
"""
import time
import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

# The authors' conv9 benchmark layer: a ResNet-style 3x3 layer, input
# (256, 64, 56, 56), 64 filters, stride 1.
B, C, H, W, CO, K, S = 256, 64, 56, 56, 64, 3, 1
# conv9 on an RTX 3090, best of thirty iterations in the sweep harness
# (results/logs/measurements/im2win_conv9_best_mean_30it.log: 37.0375); the
# authors' own harness, best of fifty, gave 37.1
# (im2win_conv9_authors_harness_50it.log). A constant, not measured here.
IM2WIN_TFLOPS = 37.0


def timeit(fn, n=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def main():
    x = torch.randn(B, C, H, W, device="cuda", dtype=torch.half)
    w = torch.randn(CO, C, K, K, device="cuda", dtype=torch.half)
    out_h = (H - K) // S + 1
    ops = 2 * B * CO * out_h * out_h * C * K * K

    # cuDNN, measured three ways because they disagree and the disagreement matters.
    # Their harness uses a host clock with a device sync per iteration, so that
    # is the methodologically matched comparison.
    host = []
    for _ in range(30):
        t0 = time.perf_counter()
        F.conv2d(x, w, stride=S)
        torch.cuda.synchronize()
        host.append(time.perf_counter() - t0)
    ev = []
    for _ in range(30):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        F.conv2d(x, w, stride=S)
        e.record()
        torch.cuda.synchronize()
        ev.append(s.elapsed_time(e) / 1000)
    b2b = timeit(lambda: F.conv2d(x, w, stride=S))

    print("cuDNN on the authors' conv9 shape")
    print(f"  host clock + sync per iter (their method): {ops/min(host)/1e12:5.2f} TFLOPS")
    print(f"  CUDA events per iter                     : {ops/min(ev)/1e12:5.2f} TFLOPS")
    print(f"  back to back, single sync                : {ops/b2b/1e12:5.2f} TFLOPS")
    # The ratios below use the most favorable of the three cuDNN figures.
    cudnn_ms = min(min(host), min(ev), b2b) * 1000

    # The windowed transform, implemented on the GPU since the released one is host-side.
    win_row = (H - K) // S + 1

    def transform():
        u = x.unfold(2, K, S)                      # (B,C,win_row,W,k)
        return u.permute(0, 1, 2, 4, 3).reshape(B, C, win_row, K * W).contiguous()

    materialized = transform()
    t_xform = timeit(transform)
    expansion = materialized.numel() / x.numel()
    traffic = (x.numel() + materialized.numel()) * 2

    kernel_ms = (ops / 1e12 / IM2WIN_TFLOPS) * 1000

    print()
    print(f"window expansion measured {expansion:.2f}x, formula predicts {(win_row*K)/H:.2f}x")
    print(f"GPU transform: {t_xform*1000:.3f} ms at {traffic/t_xform/1e9:.0f} GB/s")
    print()
    print(f"  im2win kernel alone      : {kernel_ms:6.3f} ms  ({IM2WIN_TFLOPS:.1f} TFLOPS)")
    print(f"  + materialization        : {t_xform*1000:6.3f} ms")
    print(f"  = windowed total         : {kernel_ms + t_xform*1000:6.3f} ms")
    print(f"  cuDNN, single fused call : {cudnn_ms:6.3f} ms")
    print()
    print(f"  ratio as published (kernel only): {cudnn_ms/kernel_ms:.2f}x")
    print(f"  ratio with full accounting      : {cudnn_ms/(kernel_ms + t_xform*1000):.2f}x")


if __name__ == "__main__":
    main()
