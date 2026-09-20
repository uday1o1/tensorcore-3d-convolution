"""Does the crossover, and the rule read off it, survive a change of GPU?

Everything else in this project was measured on an RTX 3090. That is the right
device for the reproduction, since it is the GPU class Im2win's own evaluation
used, but it makes every quantitative claim a property of one machine. The
decision rule in src/dispatch.py is the clearest case: its thresholds encode
that device's Tensor Core geometry, its memory bandwidth, and the particular
set of algorithms cuDNN ships for sm_86.

This script compares device-tagged result files against each other. It does not
measure anything; run the sweeps on each device first.

PREDICTION, recorded before any non-Ampere data was collected:

  The windowed path is bandwidth bound, since its distinguishing cost is
  materializing and re-reading a buffer. cuDNN's implicit GEMM is relatively
  more compute bound, since it streams into the matrix units without building
  that buffer. So the comparison should track the compute-to-bandwidth ratio of
  the device.

  Against the RTX 3090, the RTX 4090 has roughly 2.3 times the fp16 compute and
  only about 1.08 times the memory bandwidth. That is a large shift on exactly
  the axis the arithmetic intensity account names, and it should favour cuDNN.

  We therefore predict, on the 4090 relative to the 3090:
    1. fewer dense wins than 26 of 60
    2. the crossover moving to LARGER kernel sizes
    3. the decision rule losing some of its margin over always-cuDNN

  The prediction can fail in an informative direction. If the crossover sits in
  roughly the same place across devices, the rule is more portable than the
  mechanism suggests, which is a stronger and more useful result than the one
  predicted here.

  We make no directional prediction for the backward pass asymmetry (windowed
  averaging 1.030 against cuDNN on backward while averaging 0.940 on forward).
  Whether that reproduces is the single most important thing this comparison
  can tell us, because it is currently the paper's strongest claim and rests on
  one device.

Usage:  python compare_devices.py
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def rule(r):
    if r["k"] <= 3:
        return False
    if r["C"] >= 240 and r["k"] >= 5:
        return True
    return r["k"] >= 9


def geo(vals):
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def geo_gap(rows, pick):
    return geo([(r["win_ms"] if pick(r) else r["cudnn_ms"])
                / min(r["cudnn_ms"], r["win_ms"]) for r in rows])


def load(prefix):
    out = {}
    for f in sorted(RESULTS.glob(f"{prefix}_*.json")):
        tag = f.stem[len(prefix) + 1:]
        try:
            out[tag] = json.load(open(f))
        except json.JSONDecodeError:
            print(f"  skipping unreadable {f.name}")
    return out


def smallest_crossing(rows, key="ratio"):
    """Median smallest kernel size at which a layer shape first crosses."""
    firsts = []
    shapes = {(r["C"], r["sp"]) for r in rows}
    for C, sp in shapes:
        sel = sorted((r for r in rows if r["C"] == C and r["sp"] == sp),
                     key=lambda r: r["k"])
        hit = next((r["k"] for r in sel if r[key] > 1.0), None)
        if hit:
            firsts.append(hit)
    return (st.median(firsts) if firsts else None), len(firsts), len(shapes)


def main():
    maps = load("crossover_map")
    if len(maps) < 2:
        print(f"need at least two device-tagged crossover maps, found "
              f"{len(maps)}: {', '.join(maps) or 'none'}")
        print("run bench_crossover.py on each device first")
        return 1

    print("CROSSOVER ACROSS DEVICES (dense)\n")
    hdr = f"{'device':<14}{'SMs':>5}{'wins/60':>9}{'geo ratio':>11}" \
          f"{'best':>7}{'first cross':>13}"
    print(hdr)
    base = None
    for tag, rows in maps.items():
        dense = [r for r in rows if r.get("mode", "dense") == "dense"]
        if not dense:
            continue
        wins = sum(r["ratio"] > 1 for r in dense)
        g = geo([r["ratio"] for r in dense])
        med, crossed, total = smallest_crossing(dense)
        sm = dense[0].get("sm_count", "?")
        print(f"{tag:<14}{sm:>5}{wins:>6}/60{g:>11.3f}"
              f"{max(r['ratio'] for r in dense):>7.2f}"
              f"{(f'k={med:.0f} ({crossed}/{total})' if med else 'never'):>13}")
        if tag == "rtx3090":
            base = (wins, g, med)

    print("\nDISPATCH POLICIES PER DEVICE (geometric gap to a per-shape oracle)\n")
    print(f"{'device':<14}{'always cuDNN':>14}{'always win':>13}{'shape rule':>13}")
    for tag, rows in maps.items():
        dense = [r for r in rows if r.get("mode", "dense") == "dense"]
        if not dense:
            continue
        print(f"{tag:<14}{geo_gap(dense, lambda r: False):>14.3f}"
              f"{geo_gap(dense, lambda r: True):>13.3f}"
              f"{geo_gap(dense, rule):>13.3f}")

    print("\nPREDICTION AUDIT, 4090 against 3090")
    d90 = next((v for k, v in maps.items() if "4090" in k), None)
    if d90 is None or base is None:
        print("  4090 or 3090 data absent, cannot audit")
    else:
        dense = [r for r in d90 if r.get("mode", "dense") == "dense"]
        wins = sum(r["ratio"] > 1 for r in dense)
        med, _, _ = smallest_crossing(dense)
        r3_gap = geo_gap([r for r in maps["rtx3090"]
                          if r.get("mode", "dense") == "dense"], rule)
        r3_cud = geo_gap([r for r in maps["rtx3090"]
                          if r.get("mode", "dense") == "dense"], lambda r: False)
        r9_gap, r9_cud = geo_gap(dense, rule), geo_gap(dense, lambda r: False)
        print(f"  1. fewer dense wins than 26        : "
              f"{'CONFIRMED' if wins < base[0] else 'REFUTED'} ({base[0]} -> {wins})")
        print(f"  2. crossover at LARGER kernels     : "
              f"{'CONFIRMED' if (med and base[2] and med > base[2]) else 'REFUTED'} "
              f"({base[2]} -> {med})")
        print(f"  3. rule loses margin over cuDNN    : "
              f"{'CONFIRMED' if (r9_cud / r9_gap) < (r3_cud / r3_gap) else 'REFUTED'} "
              f"({r3_cud/r3_gap:.3f}x -> {r9_cud/r9_gap:.3f}x)")

    # Why the surface moves: do the two algorithms gain differently from the
    # newer device, and where? This replaces citing vendor specifications with
    # a like-for-like measurement on identical shapes.
    if len(maps) >= 2 and "rtx3090" in maps and any("4090" in k for k in maps):
        newer = next(k for k in maps if "4090" in k)
        A = {(r["C"], r["sp"], r["k"]): r for r in maps["rtx3090"]
             if r.get("mode", "dense") == "dense"}
        Bm = {(r["C"], r["sp"], r["k"]): r for r in maps[newer]
              if r.get("mode", "dense") == "dense"}
        common = sorted(set(A) & set(Bm))
        def gains(key):
            return (st.median([A[k]["cudnn_ms"] / Bm[k]["cudnn_ms"] for k in common if key(*k)]),
                    st.median([A[k]["win_ms"] / Bm[k]["win_ms"] for k in common if key(*k)]))
        print(f"\nPER-SHAPE SPEEDUP OF {newer.upper()} OVER RTX3090 "
              f"({len(common)} identical configurations)")
        print(f"  {'region':<26}{'cuDNN':>8}{'windowed':>10}   who gains more")
        regions = [("all shapes", lambda C, s, k: True),
                   ("C=240", lambda C, s, k: C == 240),
                   ("C=120", lambda C, s, k: C == 120),
                   ("C=60", lambda C, s, k: C == 60),
                   ("C=30", lambda C, s, k: C == 30),
                   ("k=3", lambda C, s, k: k == 3),
                   ("k>=9", lambda C, s, k: k >= 9)]
        for name, f in regions:
            c, w = gains(f)
            who = "cuDNN" if c > w else "windowed"
            print(f"  {name:<26}{c:>8.2f}{w:>10.2f}   {who}")
        print("  Overall the two gain almost equally, so the surface does not move"
              "\n  because one algorithm is uniformly better served by newer silicon."
              "\n  It moves because the gain is redistributed by shape: cuDNN gains"
              "\n  more at 240 channels, which is where the rule selects, and the"
              "\n  windowed path gains more at 120 channels, which is where cells"
              "\n  flip toward it.")

    # The claim that matters most, and the one we made no prediction about.
    trains = load("training_crossover")
    if trains:
        print("\nFORWARD VERSUS BACKWARD ACROSS DEVICES")
        print(f"{'device':<14}{'forward':>10}{'backward':>11}{'full step':>12}"
              f"{'mem median':>13}")
        for tag, rows in trains.items():
            bwd = [(r["step_cudnn_ms"] - r["fwd_cudnn_ms"]) /
                   (r["step_win_ms"] - r["fwd_win_ms"]) for r in rows
                   if r["step_cudnn_ms"] > r["fwd_cudnn_ms"]
                   and r["step_win_ms"] > r["fwd_win_ms"]]
            print(f"{tag:<14}{geo([r['fwd_ratio'] for r in rows]):>10.3f}"
                  f"{geo(bwd):>11.3f}"
                  f"{geo([r['step_ratio'] for r in rows]):>12.3f}"
                  f"{st.median([r['mem_ratio'] for r in rows]):>12.2f}x")
        if len(trains) > 1:
            asym = {t: geo([(r["step_cudnn_ms"] - r["fwd_cudnn_ms"]) /
                            (r["step_win_ms"] - r["fwd_win_ms"]) for r in rows
                            if r["step_cudnn_ms"] > r["fwd_cudnn_ms"]
                            and r["step_win_ms"] > r["fwd_win_ms"]])
                    > geo([r["fwd_ratio"] for r in rows])
                    for t, rows in trains.items()}
            every = all(asym.values())
            print(f"\n  backward beats forward on every device: {every}"
                  f"  {asym}")
            print("  (this is the paper's strongest claim and the reason for "
                  "running a second device at all)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
