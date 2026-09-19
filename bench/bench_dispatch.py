"""Tests the decision rule on shapes it was never derived from.

The rule in src/dispatch.py was read off the crossover grid. Evaluating it on
that same grid would only measure how well it describes its own training data,
which is not a claim about anything. This benchmark evaluates it on a disjoint
set of shapes: channel widths, spatial extents, and kernel sizes that do not
appear in the grid at all.

Four policies are compared against a per-shape oracle that always picks the
faster of the two algorithms after the fact:

  cudnn     always implicit GEMM, the incumbent
  windowed  always the windowed method, what the literature implicitly proposes
  rule      decide from shape alone, at zero runtime cost
  autotune  time both once per shape and keep the winner

The headline metric is the geometric mean of (policy time / oracle time) over
shapes. The arithmetic total is reported too but is the wrong summary here:
it is dominated by the few largest configurations, so a policy can look good
on it while being wrong about most layers.

Correctness is not re-checked here; run verify_correctness.py first.

Usage:  python bench_dispatch.py [trials]
"""
import json
import math
import statistics as st
import sys
from pathlib import Path
import time

import torch
import torch.nn.functional as F

# Resolve paths from this file, not the working directory, so the script
# runs from anywhere in a fresh clone.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from winconv import im2win_conv3d_minmat
from dispatch import prefer_windowed

torch.backends.cudnn.benchmark = True

# Deliberately disjoint from bench_crossover.py's grid, which uses
# C in {30,60,120,240}, spatial in {16,32,64}, k in {3,5,7,9,11}.
CHANNELS = [48, 96, 192, 320]
SPATIAL = [24, 48]
KERNELS = [3, 5, 7, 9]
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


def preflight():
    """Check the autotuner reproduces decisions the grid already settled.

    The autotuner is used as the oracle the rule is scored against, so it has
    to be trustworthy before any of the held-out numbers mean anything. These
    five shapes are in the measured grid, with known and unambiguous winners.
    """
    from dispatch import DispatchingConv3d
    known = {(60, 64, 3): 0.63, (240, 64, 5): 1.45, (240, 16, 3): 0.84,
             (120, 64, 9): 1.28, (30, 32, 5): 0.62}
    print("preflight: autotuner against shapes the grid already settled")
    hdr = ("shape", "grid", "truth", "autotune", "agree")
    print(f"  {hdr[0]:>14}{hdr[1]:>8}{hdr[2]:>8}{hdr[3]:>10}{hdr[4]:>7}")
    ok = True
    for (C, sp, k), ratio in known.items():
        m = DispatchingConv3d(C, C, k, policy="autotune").cuda().half()
        x = torch.randn(BATCH, C, sp, sp, sp, device="cuda", dtype=torch.half)
        m(x)
        got = list(m._choice.values())[0]
        truth = ratio > 1.0
        ok &= got == truth
        label = f"C{C} sp{sp} k{k}"
        print(f"  {label:>14}{ratio:8.2f}{str(truth):>8}{str(got):>10}"
              f"{str(got == truth):>7}")
        del m, x
        torch.cuda.empty_cache()
    print(f"  autotuner trustworthy: {ok}\n")
    return ok


def main(trials=5):
    if not preflight():
        print("ABORT: the autotuner disagrees with already-measured shapes, so it "
              "cannot serve as the oracle. Fix its timing before trusting anything below.")
        return 1
    rows = []
    print(f"held-out grid: C={CHANNELS} spatial={SPATIAL} k={KERNELS} batch={BATCH}")
    print(f'{"C":>5}{"sp":>4}{"k":>3}{"out":>5}{"cuDNN_ms":>10}{"win_ms":>9}'
          f'{"ratio":>7}  {"rule":>5} {"correct":>8}')
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
                        pairs.append((timeit(lambda: F.conv3d(x, w)),
                                      timeit(lambda: im2win_conv3d_minmat(x, w))))
                    tc = st.median([a for a, _ in pairs])
                    tw = st.median([b for _, b in pairs])
                    said = prefer_windowed(C, sp, k)
                    truth = tw < tc
                    rows.append({"C": C, "sp": sp, "k": k, "out": out,
                                 "batch": BATCH, "trials": trials,
                                 "cudnn_ms": tc * 1000, "win_ms": tw * 1000,
                                 "ratio": tc / tw, "rule_says_windowed": said,
                                 "windowed_actually_faster": truth})
                    print(f"{C:5d}{sp:4d}{k:3d}{out:5d}{tc*1000:10.2f}{tw*1000:9.2f}"
                          f"{tc/tw:7.2f}  {str(said):>5} {str(said==truth):>8}",
                          flush=True)
                    del x, w
                    torch.cuda.empty_cache()
                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    torch.cuda.empty_cache()
                    print(f"{C:5d}{sp:4d}{k:3d}{out:5d}      oom", flush=True)

    if not rows:
        print("no configurations measured")
        return 1
    json.dump(rows, open(ROOT / "results" / "dispatch_heldout.json", "w"), indent=1)

    oracle = sum(min(r["cudnn_ms"], r["win_ms"]) for r in rows)

    def report(name, pick):
        total = sum((r["win_ms"] if pick(r) else r["cudnn_ms"]) for r in rows)
        geo = math.exp(sum(math.log((r["win_ms"] if pick(r) else r["cudnn_ms"])
                                    / min(r["cudnn_ms"], r["win_ms"]))
                           for r in rows) / len(rows))
        print(f"  {name:10s} total {total:9.1f} ms   oracle-gap {total/oracle:5.3f}x"
              f"   geo-gap {geo:5.3f}x")
        return geo

    n = len(rows)
    agree = sum(r["rule_says_windowed"] == r["windowed_actually_faster"] for r in rows)
    wins = sum(r["windowed_actually_faster"] for r in rows)
    print(f"\nheld-out shapes: {n}, windowed actually faster in {wins}")
    print(f"rule agrees with ground truth in {agree}/{n} ({agree/n:.1%})")

    print("\npolicy comparison, lower gap is better:")
    g_cud = report("cudnn", lambda r: False)
    g_win = report("windowed", lambda r: True)
    g_rule = report("rule", lambda r: r["rule_says_windowed"])
    report("oracle", lambda r: r["windowed_actually_faster"])

    print(f"\nrule vs always-cuDNN:    {g_cud/g_rule:.3f}x better")
    print(f"rule vs always-windowed: {g_win/g_rule:.3f}x better")

    # The rule is only worth stating if it beats both pure strategies.
    verdict = "HOLDS" if (g_rule < g_cud and g_rule < g_win) else "FAILS"
    print(f"\nclaim 'dispatch beats both pure strategies on unseen shapes': {verdict}")

    # Where the rule is wrong, how much does it cost?
    bad = [r for r in rows if r["rule_says_windowed"] != r["windowed_actually_faster"]]
    if bad:
        print(f"\n{len(bad)} disagreements:")
        for r in sorted(bad, key=lambda r: -abs(r["cudnn_ms"] - r["win_ms"]))[:6]:
            cost = (r["win_ms"] - r["cudnn_ms"]) if r["rule_says_windowed"] \
                else (r["cudnn_ms"] - r["win_ms"])
            print(f"  C{r['C']:3d} sp{r['sp']:2d} k{r['k']:2d}: rule said "
                  f"{'windowed' if r['rule_says_windowed'] else 'cudnn':8s}, "
                  f"ratio {r['ratio']:.2f}, cost {cost:+.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 5))
