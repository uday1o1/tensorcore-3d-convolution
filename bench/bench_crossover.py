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
import sys, json, time, statistics as st
import torch
import torch.nn.functional as F

sys.path.insert(0, "../src")
from winconv import im2win_conv3d_minmat

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


def main(trials=3):
    rows = []
    print(f'{"C":>5}{"sp":>5}{"k":>4}{"out":>5}{"cuDNN_ms":>10}{"win_ms":>9}{"ratio":>8}')
    for C in CHANNELS:
        for sp in SPATIAL:
            for k in KERNELS:
                out = sp - k + 1
                if out < 4:
                    continue
                try:
                    x = torch.randn(BATCH, C, sp, sp, sp, device="cuda", dtype=torch.half)
                    w = torch.randn(C, C, k, k, k, device="cuda", dtype=torch.half)
                    pairs = []
                    for _ in range(trials):
                        tc = timeit(lambda: F.conv3d(x, w))
                        tw = timeit(lambda: im2win_conv3d_minmat(x, w))
                        pairs.append((tc, tw))
                    ratios = [a / b for a, b in pairs]
                    tc = st.median([a for a, _ in pairs])
                    tw = st.median([b for _, b in pairs])
                    rows.append({"C": C, "sp": sp, "k": k, "out": out,
                                 "batch": BATCH, "trials": trials,
                                 "cudnn_ms": tc * 1000, "win_ms": tw * 1000,
                                 "ratio": tc / tw,
                                 "ratio_min": min(ratios), "ratio_max": max(ratios)})
                    print(f"{C:5d}{sp:5d}{k:4d}{out:5d}{tc*1000:10.2f}{tw*1000:9.2f}"
                          f"{tc/tw:8.2f}  [{min(ratios):.2f},{max(ratios):.2f}]", flush=True)
                    del x, w
                    torch.cuda.empty_cache()
                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    torch.cuda.empty_cache()
                    print(f"{C:5d}{sp:5d}{k:4d}{out:5d}       oom", flush=True)
    json.dump(rows, open("../results/crossover_map.json", "w"), indent=1)
    wins = [r for r in rows if r["ratio"] > 1.0]
    print(f"\nwindowed wins in {len(wins)}/{len(rows)} configurations")
    if wins:
        best = max(wins, key=lambda r: r["ratio"])
        print(f"best: C={best['C']} sp={best['sp']} k={best['k']} -> {best['ratio']:.2f}x")

    # The two rules the paper states as holding without exception, checked
    # against the WORST trial in each cell rather than the median.
    for label, sel in (("never wins at k=3", [r for r in rows if r["k"] == 3]),
                       ("always wins at C>=240, k>=5",
                        [r for r in rows if r["C"] >= 240 and r["k"] >= 5])):
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
