"""
Maps where windowed convolution beats cuDNN, across channel width, spatial
extent, and kernel size, in 3D.

This produces the paper's central positive result: implicit GEMM scales with
the k^3 growth in work while windowed materialization grows only linearly in
k, so the curves cross at large kernel sizes.

Reports the MEDIAN of repeated trials, and also the per-trial ratio spread.
cuDNN shows real run-to-run variance in algorithm selection, and single
measurements here are not trustworthy: three separate headline numbers in this
project had to be corrected after repeated measurement contradicted a
single-shot reading.

The spread is recorded because the paper states rules that hold "without
exception" over this grid. A median above 1.0 does not establish that; a cell
whose worst trial is also above 1.0 does. `ratio_min` is what those claims
are checked against.

Usage:  python bench_crossover.py [trials]
"""
import json
import statistics as st
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F

# Resolve paths from this file, not the working directory, so the script
# runs from anywhere in a fresh clone.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from winconv import im2win_conv3d_minmat, im2win_conv3d_depthwise
from benchenv import (results_dir, device_tag, device_meta, timed_pair,
                      order_summary, discard_warmup)

torch.backends.cudnn.benchmark = True

CHANNELS = [30, 60, 120, 240]
SPATIAL = [16, 32, 64]
KERNELS = [3, 5, 7, 9, 11]
BATCH = 2




def timeit(fn, n=8):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def sweep(mode, trials, rows, meta):
    """mode is 'dense' (groups=1) or 'depthwise' (groups=C)."""
    print(f'\n=== {mode} ===')
    print(f'{"C":>5}{"sp":>5}{"k":>4}{"out":>5}{"cuDNN_ms":>10}{"win_ms":>9}'
          f'{"ratio":>8}  spread')
    for C in CHANNELS:
        for sp in SPATIAL:
            for k in KERNELS:
                out = sp - k + 1
                if out < 4:
                    continue
                try:
                    x = torch.randn(BATCH, C, sp, sp, sp, device="cuda", dtype=torch.half)
                    if mode == "dense":
                        w = torch.randn(C, C, k, k, k, device="cuda", dtype=torch.half)
                        ref = lambda: F.conv3d(x, w)
                        win = lambda: im2win_conv3d_minmat(x, w)
                    else:
                        w = torch.randn(C, 1, k, k, k, device="cuda", dtype=torch.half)
                        ref = lambda: F.conv3d(x, w, groups=C)
                        win = lambda: im2win_conv3d_depthwise(x, w)
                    # Order alternates by trial so a position effect cancels in
                    # the median and is recorded rather than assumed absent.
                    pairs = [timed_pair(ref, win, timeit, t) for t in range(trials)]
                    ratios = [a / b for a, b, _ in pairs]
                    tc = st.median([a for a, _, _ in pairs])
                    tw = st.median([b for _, b, _ in pairs])
                    rows.append({**meta, "mode": mode, "C": C, "sp": sp, "k": k, "out": out,
                                 "batch": BATCH, "trials": trials,
                                 "cudnn_ms": tc * 1000, "win_ms": tw * 1000,
                                 "ratio": tc / tw,
                                 "ratio_min": min(ratios), "ratio_max": max(ratios),
                                 **order_summary(pairs)})
                    oe = order_summary(pairs).get("order_effect")
                    print(f"{C:5d}{sp:5d}{k:4d}{out:5d}{tc*1000:10.2f}{tw*1000:9.2f}"
                          f"{tc/tw:8.2f}  [{min(ratios):.2f},{max(ratios):.2f}]"
                          + (f"  order {oe:.2f}" if oe else ""), flush=True)
                    del x, w
                    torch.cuda.empty_cache()
                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    torch.cuda.empty_cache()
                    print(f"{C:5d}{sp:5d}{k:4d}{out:5d}       oom", flush=True)


def main(trials=3):
    rows = []
    meta = device_meta()
    print({k: v for k, v in meta.items() if k in ("gpu", "driver", "power_limit_w",
                                                  "power_default_w", "host_cpu")})
    # The very first measurement of a sequence is penalised; spend it here.
    x0 = torch.randn(BATCH, 30, 16, 16, 16, device="cuda", dtype=torch.half)
    w0 = torch.randn(30, 30, 3, 3, 3, device="cuda", dtype=torch.half)
    discard_warmup(lambda: F.conv3d(x0, w0))
    discard_warmup(lambda: im2win_conv3d_minmat(x0, w0))
    del x0, w0
    sweep("dense", trials, rows, meta)
    sweep("depthwise", trials, rows, meta)
    json.dump(rows, open(results_dir() / f"crossover_map_{device_tag()}.json", "w"), indent=1)

    for mode in ("dense", "depthwise"):
        sel = [r for r in rows if r["mode"] == mode]
        if not sel:
            continue
        wins = [r for r in sel if r["ratio"] > 1.0]
        print(f"\n{mode}: windowed wins in {len(wins)}/{len(sel)} configurations")
        if wins:
            best = max(wins, key=lambda r: r["ratio"])
            print(f"  best: C={best['C']} sp={best['sp']} k={best['k']} "
                  f"-> {best['ratio']:.2f}x")

    # The two rules the paper states as holding in every configuration
    # measured, checked against the WORST trial in each cell, not the median.
    dense = [r for r in rows if r["mode"] == "dense"]
    for label, sel in (("never wins at k=3", [r for r in dense if r["k"] == 3]),
                       ("always wins at C>=240, k>=5",
                        [r for r in dense if r["C"] >= 240 and r["k"] >= 5])):
        if not sel:
            continue
        if "never" in label:
            ok = all(r["ratio_max"] < 1.0 for r in sel)
            edge = max(r["ratio_max"] for r in sel)
        else:
            ok = all(r["ratio_min"] > 1.0 for r in sel)
            edge = min(r["ratio_min"] for r in sel)
        print(f"rule '{label}': {'HOLDS' if ok else 'VIOLATED'} "
              f"over {len(sel)} cells, worst trial {edge:.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
