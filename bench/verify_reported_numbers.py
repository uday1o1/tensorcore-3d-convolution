"""Recomputes every derived number reported in the write-up from results/.

All reported numbers come from the RTX 3090, the GPU class Im2win's own
evaluation used. Other devices write their own device-tagged files; this checks
the ones the write-up quotes.

The measured data in results/ is raw: per-configuration timings. Most of the
claims made about it are derived, things like win counts, medians, geometric
means against an oracle, and rule agreement rates. A reader has no way to check
those by eye, and a transcription error between the data and the prose would be
invisible.

This script recomputes each of them from the committed JSON and prints them
next to the value the write-up states, so any drift shows up as MISMATCH. It
needs no GPU and no dataset: it reads results/ and nothing else.

Run it after any re-measurement. If a number here changes, the prose has to
change with it.

Usage:  python verify_reported_numbers.py
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# What the write-up claims. Update these together with the prose, never alone.
#
# Counts must match exactly. Rounded measurements are compared within a
# relative tolerance, since the prose quotes them to a couple of significant
# figures and the data carries full precision.
COUNTS = {
    "dense configurations won", "depthwise configurations won",
    "wins at k>=9", "wins at C>=240 and k>=5",
    "held out, rule agreements out of 32",
}
CLAIMED = {
    "dense configurations won": 26,
    "depthwise configurations won": 2,
    "dense median ratio": 0.95,
    "depthwise median ratio": 0.84,
    "best dense ratio": 2.36,
    "best depthwise ratio": 1.03,
    "wins at k>=9": 19,
    "wins at C>=240 and k>=5": 11,
    "largest single-layer saving, ms": 735,
    "geo gap, always cuDNN": 1.132,
    "geo gap, always windowed": 1.160,
    "geo gap, shape rule": 1.021,
    "held out, geo gap always cuDNN": 1.126,
    "held out, geo gap always windowed": 1.098,
    "held out, geo gap shape rule": 1.041,
    "held out, rule agreements out of 32": 24,
    "training, geo ratio forward": 0.940,
    "training, geo ratio backward": 1.030,   # rtx3090 only; 0.937 on rtx4090
    "training, geo ratio full step": 1.015,
    "training, peak memory median": 2.16,
}


def rule(r):
    """The decision rule from src/dispatch.py, restated for independence."""
    if r["k"] <= 3:
        return False
    if r["C"] >= 240 and r["k"] >= 5:
        return True
    return r["k"] >= 9


def geo_gap(rows, pick):
    """Geometric mean of (policy time / best possible time) over rows."""
    return math.exp(sum(
        math.log((r["win_ms"] if pick(r) else r["cudnn_ms"])
                 / min(r["cudnn_ms"], r["win_ms"])) for r in rows) / len(rows))


def main():
    cm = ROOT / "results" / "crossover_map_rtx3090.json"
    ho = ROOT / "results" / "dispatch_heldout_rtx3090.json"
    missing = [p.name for p in (cm, ho) if not p.exists()]
    if missing:
        print(f"missing results files: {', '.join(missing)}")
        print("regenerate with bench_crossover.py and bench_dispatch.py")
        return 1

    tc = ROOT / "results" / "training_crossover_rtx3090.json"
    allrows = json.load(open(cm))
    dense = [r for r in allrows if r.get("mode", "dense") == "dense"]
    depth = [r for r in allrows if r.get("mode") == "depthwise"]
    held = json.load(open(ho))

    train = json.load(open(tc)) if tc.exists() else []

    def geo(vals):
        return math.exp(sum(math.log(v) for v in vals) / len(vals))

    big = max(dense, key=lambda r: r["cudnn_ms"] - r["win_ms"])
    computed = {
        "dense configurations won": sum(r["ratio"] > 1 for r in dense),
        "depthwise configurations won": sum(r["ratio"] > 1 for r in depth),
        "dense median ratio": st.median([r["ratio"] for r in dense]),
        "depthwise median ratio": st.median([r["ratio"] for r in depth]),
        "best dense ratio": max(r["ratio"] for r in dense),
        "best depthwise ratio": max(r["ratio"] for r in depth),
        "wins at k>=9": sum(r["ratio"] > 1 for r in dense if r["k"] >= 9),
        "wins at C>=240 and k>=5": sum(r["ratio"] > 1 for r in dense
                                       if r["C"] >= 240 and r["k"] >= 5),
        "largest single-layer saving, ms": big["cudnn_ms"] - big["win_ms"],
        "geo gap, always cuDNN": geo_gap(dense, lambda r: False),
        "geo gap, always windowed": geo_gap(dense, lambda r: True),
        "geo gap, shape rule": geo_gap(dense, rule),
        "held out, geo gap always cuDNN": geo_gap(held, lambda r: False),
        "held out, geo gap always windowed": geo_gap(held, lambda r: True),
        "held out, geo gap shape rule": geo_gap(held, lambda r: r["rule_says_windowed"]),
        "held out, rule agreements out of 32":
            sum(r["rule_says_windowed"] == r["windowed_actually_faster"] for r in held),
    }
    if train:
        bwd = [(r["step_cudnn_ms"] - r["fwd_cudnn_ms"]) /
               (r["step_win_ms"] - r["fwd_win_ms"]) for r in train
               if r["step_cudnn_ms"] > r["fwd_cudnn_ms"]
               and r["step_win_ms"] > r["fwd_win_ms"]]
        computed.update({
            "training, geo ratio forward": geo([r["fwd_ratio"] for r in train]),
            "training, geo ratio backward": geo(bwd),
            "training, geo ratio full step": geo([r["step_ratio"] for r in train]),
            "training, peak memory median": st.median([r["mem_ratio"] for r in train]),
        })
    else:
        for k in list(CLAIMED):
            if k.startswith("training,"):
                CLAIMED.pop(k)
        print("note: results/training_crossover.json absent, skipping training claims\n")

    print(f"crossover grid: {len(dense)} dense, {len(depth)} depthwise, "
          f"{dense[0].get('trials','?')} trials each")
    print(f"held out grid:  {len(held)} configurations\n")
    print(f"  {'claim':<38}{'stated':>10}{'computed':>12}")
    ok = True
    for name, claim in CLAIMED.items():
        got = computed[name]
        tol = 0 if name in COUNTS else 0.015 * abs(claim)
        good = abs(got - claim) <= tol
        ok &= good
        print(f"  {name:<38}{claim:>10}{got:>12.3f}  "
              f"{'' if good else '<-- MISMATCH'}")

    # The rule as coded must match the rule as recorded in the held-out run.
    drift = [r for r in held if rule(r) != r["rule_says_windowed"]]
    if drift:
        ok = False
        print(f"\n  MISMATCH: the rule in this file disagrees with the one used "
              f"during the held-out run on {len(drift)} configurations")

    print("\n" + ("ALL REPORTED NUMBERS MATCH THE DATA" if ok
                  else "DRIFT BETWEEN THE WRITE-UP AND THE DATA"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
