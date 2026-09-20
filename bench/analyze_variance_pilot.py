"""Is the kernel-placement comparison measurable at this budget?

Two identically configured MedNeXt-B kernel-3 runs differ only in random seed.
The spread between them is the noise floor of this training setup. Any claim
that one kernel placement beats another has to clear that floor.

The threshold is not arbitrary. MedNeXt reports that uniform kernel 5 with
UpKern improves mean Dice by roughly 0.2 points over kernel 3 (84.03 to 84.23
on BTCV, 86.71 to 87.06 on AMOS22). That is the size of the effect a placement
study is chasing, so if two identical runs differ by more than that, single-run
comparisons between placements cannot be interpreted.

The verdict is computed here rather than judged by eye, and the threshold was
fixed before the runs finished.

Outcomes:
  GO        spread is well below the effect; single runs per placement suffice
  MARGINAL  spread is comparable to the effect; repeats are needed
  NO GO     spread exceeds the effect; the accuracy axis is not measurable
            here, and the cost-side result stands on its own instead

Usage:  python analyze_variance_pilot.py [log1] [log2]
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The effect the study is trying to resolve, from MedNeXt's own reported gain
# for uniform kernel 5 with UpKern over kernel 3.
TARGET_EFFECT = 0.2 / 100.0   # 0.2 Dice points, on a 0-1 scale

DICE_RE = re.compile(r"Average global foreground Dice: \[([^\]]*)\]")
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def dice_curve(path):
    """Per-epoch validation Dice for each foreground class."""
    out = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = DICE_RE.search(line)
        if m:
            vals = [float(v) for v in NUM_RE.findall(m.group(1))]
            if vals:
                out.append(vals)
    return out


def final_dice(curve, tail=5):
    """Mean over the last few epochs, to damp single-epoch fluctuation."""
    if not curve:
        return None
    n = min(tail, len(curve))
    ncls = len(curve[-1])
    return [st.mean([c[i] for c in curve[-n:] if len(c) > i]) for i in range(ncls)]


def main(argv):
    logs = argv[1:3] or ["/workspace/pilot_s1.log", "/workspace/pilot_s2.log"]
    curves, finals = [], []
    for p in logs:
        if not Path(p).exists():
            print(f"missing log: {p}")
            print("run both variance pilot seeds first")
            return 1
        c = dice_curve(p)
        curves.append(c)
        finals.append(final_dice(c))
        print(f"{p}: {len(c)} validation points, final Dice {finals[-1]}")

    if any(f is None for f in finals):
        print("no Dice values parsed; check the log format")
        return 1
    if len({len(f) for f in finals}) != 1:
        print("the two runs report different class counts; not comparable")
        return 1

    ncls = len(finals[0])
    names = ["liver", "tumor"][:ncls] or [f"class{i}" for i in range(ncls)]
    print(f"\n{'class':<10}{'seed 1':>10}{'seed 2':>10}{'spread':>10}")
    spreads = []
    for i in range(ncls):
        a, b = finals[0][i], finals[1][i]
        spreads.append(abs(a - b))
        print(f"{names[i]:<10}{a:>10.4f}{b:>10.4f}{abs(a-b):>10.4f}")

    mean_spread = st.mean(spreads)
    worst = max(spreads)
    print(f"\nmean spread {mean_spread:.4f}, worst class {worst:.4f}")
    print(f"effect being chased: {TARGET_EFFECT:.4f} "
          f"(MedNeXt's reported kernel 5 gain over kernel 3)")

    if worst < TARGET_EFFECT / 2:
        verdict, note = "GO", ("spread is well under the effect; one run per "
                               "placement is interpretable")
    elif worst < TARGET_EFFECT * 1.5:
        verdict, note = "MARGINAL", ("spread is comparable to the effect; "
                                     "placements need repeats, not more configurations")
    else:
        verdict, note = "NO GO", ("spread exceeds the effect; the accuracy axis is "
                                  "not measurable at this budget. Report the cost-side "
                                  "dominance result, which needs no training, and say "
                                  "plainly why accuracy is unresolved")
    print(f"\nVERDICT: {verdict}\n  {note}")

    json.dump({"logs": logs, "final_dice": finals, "spreads": spreads,
               "mean_spread": mean_spread, "worst_spread": worst,
               "target_effect": TARGET_EFFECT, "verdict": verdict},
              open(ROOT / "results" / "variance_pilot.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
