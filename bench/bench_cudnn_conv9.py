"""cuDNN's throughput on Im2win's conv9 layer, across fresh processes.

The kernel-level comparison with Im2win rests on one cuDNN number, and cuDNN
measured on this layer has ranged from 17.6 to 42.4 TFLOPS in separate runs
(results/logs/measurements/). With cudnn.benchmark on, each process picks its
algorithm by timing candidates on first use, so two processes can settle on
different kernels. This runs the same measurement in many fresh processes and
records, for each, the throughput by three timing methods and the names of
the GPU kernels the convolution actually launched, so a fast or slow run can
be traced to the kernel that produced it.

Layer: input (256, 64, 56, 56), weight (64, 64, 3, 3), stride 1, fp16, NCHW,
the shape Im2win's conv9 benchmark uses. Throughput counts two operations per
multiply-accumulate. The Im2win kernel itself is not run here; its 37.0 TFLOPS
on an RTX 3090 is in results/logs/measurements/im2win_conv9_best_mean_30it.log.

Usage:  python bench_cudnn_conv9.py [n]   (default 20: n processes with cudnn.benchmark
        on, then max(1, n // 4) with it off)
Writes: cudnn_conv9_runs_<gpu>.json in the results directory.
"""
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

B, C, H, W, CO, K, S = 256, 64, 56, 56, 64, 3, 1
OH = (H - K) // S + 1
OPS = 2 * B * CO * OH * OH * C * K * K
IM2WIN_TFLOPS = 37.0


def child(benchmark):
    import torch
    import torch.nn.functional as F
    torch.backends.cudnn.benchmark = benchmark
    x = torch.randn(B, C, H, W, device="cuda", dtype=torch.half)
    w = torch.randn(CO, C, K, K, device="cuda", dtype=torch.half)
    for _ in range(10):
        F.conv2d(x, w, stride=S)
    torch.cuda.synchronize()

    ev = []
    for _ in range(50):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        F.conv2d(x, w, stride=S)
        b.record()
        torch.cuda.synchronize()
        ev.append(a.elapsed_time(b) / 1000)

    host = []
    for _ in range(30):
        t0 = time.perf_counter()
        F.conv2d(x, w, stride=S)
        torch.cuda.synchronize()
        host.append(time.perf_counter() - t0)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        F.conv2d(x, w, stride=S)
    torch.cuda.synchronize()
    b2b = (time.perf_counter() - t0) / 30

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        F.conv2d(x, w, stride=S)
        torch.cuda.synchronize()
    kernels = sorted({e.name for e in prof.events()
                      if e.device_type.name == "CUDA"})

    tf = lambda t: OPS / t / 1e12  # noqa: E731
    print(json.dumps({
        "benchmark": benchmark,
        "events_best_tflops": tf(min(ev)), "events_mean_tflops": tf(st.mean(ev)),
        "host_best_tflops": tf(min(host)), "back_to_back_tflops": tf(b2b),
        "kernels": kernels}))


def main(argv):
    if "--child" in argv:
        child(benchmark="--no-benchmark" not in argv)
        return 0
    import torch
    if not torch.cuda.is_available():
        print("needs a CUDA device")
        return 1
    from benchenv import results_dir, device_tag, device_meta
    n = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else 20
    runs = []
    # Fresh processes, so each makes its own algorithm choice. A few runs with
    # the autotuner off show cuDNN's default choice for comparison.
    plan = [True] * n + [False] * max(1, n // 4)
    for i, bench in enumerate(plan):
        cmd = [sys.executable, __file__, "--child"] + ([] if bench else ["--no-benchmark"])
        out = subprocess.run(cmd, capture_output=True, text=True)
        line = next((l for l in out.stdout.splitlines() if l.startswith("{")), None)
        if line is None:
            print(f"run {i}: failed\n{out.stderr[-800:]}")
            continue
        r = json.loads(line)
        r["run"] = i
        runs.append(r)
        print(f"run {i:2d} benchmark={str(bench):<5} events best "
              f"{r['events_best_tflops']:6.2f}  back to back {r['back_to_back_tflops']:6.2f}"
              f"  host best {r['host_best_tflops']:6.2f} TFLOPS  "
              f"{len(r['kernels'])} kernel(s)", flush=True)
    if not runs:
        return 1
    meta = device_meta()
    for r in runs:
        r.update(meta)
    json.dump(runs, open(results_dir() / f"cudnn_conv9_runs_{device_tag()}.json", "w"),
              indent=1)

    for bench in (True, False):
        sel = [r for r in runs if r["benchmark"] == bench]
        if not sel:
            continue
        print(f"\ncudnn.benchmark={bench}, {len(sel)} processes")
        for key in ("events_best_tflops", "back_to_back_tflops", "host_best_tflops"):
            v = [r[key] for r in sel]
            print(f"  {key:<22} min {min(v):6.2f}  median {st.median(v):6.2f}  "
                  f"max {max(v):6.2f}  (Im2win kernel / cuDNN at median "
                  f"{IM2WIN_TFLOPS / st.median(v):.2f})")
        groups = {}
        for r in sel:
            groups.setdefault(tuple(r["kernels"]), []).append(r["events_best_tflops"])
        for ks, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            print(f"  {len(v):2d} process(es), events best {min(v):.2f} to {max(v):.2f}:")
            for k in ks:
                print(f"      {k[:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
