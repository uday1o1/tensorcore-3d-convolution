"""Is the kernel-placement comparison measurable at this budget?

Two identically configured MedNeXt-B kernel-3 runs differ only in random seed.
The spread between them is the noise floor of this training setup, and any claim
that one kernel placement beats another has to clear it.

The threshold is not arbitrary. MedNeXt reports uniform kernel 5 with UpKern
improving mean Dice by roughly 0.2 points over kernel 3 (84.03 to 84.23 on BTCV,
86.71 to 87.06 on AMOS22). That is the size of the effect a placement study
chases, so if two identical runs differ by more than that, single-run
comparisons between placements cannot be interpreted.

WHICH NUMBER TO COMPARE. nnU-Net logs an "Average global foreground Dice" every
epoch, computed on training-time validation batches. It is extremely noisy: in
one of our runs consecutive epochs gave tumour Dice of 0.79, 0.63, 0.74, 0.79,
0.50, 0.84. That figure is a training diagnostic, not a result, and an earlier
version of this script compared it, which would have declared the study
impossible on the strength of the wrong quantity. The result is the per-case
Dice in validation_raw/summary.json, produced after training by a full sliding
window inference over every validation case.

Having per-case values for both runs also permits a paired test over cases
rather than a difference of two means, which is what the project proposal
specified and which is the stronger comparison.

Outcomes:
  GO        spread well below the effect; single runs per placement suffice
  MARGINAL  spread comparable to the effect; placements need repeats
  NO GO     spread exceeds the effect; the accuracy axis is not measurable at
            this budget, and the cost-side result stands on its own instead

Usage:
  python analyze_variance_pilot.py <summary1.json> <summary2.json>
  python analyze_variance_pilot.py           # uses the two pilot trainer paths
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# MedNeXt's reported gain for uniform kernel 5 over kernel 3, on a 0-1 scale.
TARGET_EFFECT = 0.002

RESULTS = Path("/workspace/RESULTS_FOLDER/nnUNet/3d_fullres/Task003_Liver")
DEFAULTS = [
    RESULTS / "nnUNetTrainerV2_MedNeXt_B_k3_150ep_s1__nnUNetPlansv2.1_trgSp_1x1x1"
            / "fold_0" / "validation_raw" / "summary.json",
    RESULTS / "nnUNetTrainerV2_MedNeXt_B_k3_150ep_s2__nnUNetPlansv2.1_trgSp_1x1x1"
            / "fold_0" / "validation_raw" / "summary.json",
]
CLASS_NAMES = {"1": "liver", "2": "tumor"}


def per_case(path):
    """Per-case Dice by class label, keyed by case identifier."""
    d = json.load(open(path))
    out = {}
    for case in d["results"]["all"]:
        ref = case.get("reference") or case.get("test") or ""
        key = Path(str(ref)).name
        out[key] = {k: v["Dice"] for k, v in case.items()
                    if k.isdigit() and isinstance(v, dict) and "Dice" in v}
    return out


def wilcoxon_signed_rank(diffs):
    """Two-sided Wilcoxon signed-rank p, normal approximation.

    Returns None when too few nonzero differences for the approximation to
    mean anything, rather than returning a number that looks like evidence.
    """
    nz = [d for d in diffs if d != 0 and not math.isnan(d)]
    n = len(nz)
    if n < 6:
        return None
    ranked = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    for pos, i in enumerate(ranked, start=1):
        ranks[i] = float(pos)
    w_plus = sum(ranks[i] for i in range(n) if nz[i] > 0)
    mean = n * (n + 1) / 4
    sd = (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    if sd == 0:
        return None
    z = (w_plus - mean) / sd
    # two-sided normal tail
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def main(argv):
    paths = [Path(p) for p in argv[1:3]] or DEFAULTS
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("missing validation summaries:")
        for m in missing:
            print(f"  {m}")
        print("\nBoth pilot seeds must finish training AND validation first.")
        print("The per-epoch Dice in the training log is a diagnostic, not a")
        print("result, and must not be substituted for these.")
        return 1

    runs = [per_case(p) for p in paths]
    shared = sorted(set(runs[0]) & set(runs[1]))
    if not shared:
        print("the two runs share no validation cases; not comparable")
        return 1
    classes = sorted({c for r in runs for v in r.values() for c in v}
                     - {"0"})

    print(f"comparing {len(shared)} shared validation cases\n")
    print(f"{'class':<10}{'seed 1':>10}{'seed 2':>10}{'spread':>10}"
          f"{'per-case sd':>13}{'paired p':>10}")
    spreads = []
    report = {}
    for c in classes:
        # A class Dice is NaN for cases that do not contain that class. Liver
        # tumour is absent from some validation volumes, so those cases carry
        # no information about this class and must be dropped rather than
        # averaged. Keeping them crashes statistics.stdev, which is how this
        # was found; silently coercing them to zero would have been worse,
        # since it would drag both means toward zero and shrink the apparent
        # spread, making the noise floor look smaller than it is.
        usable = [k for k in shared
                  if c in runs[0][k] and c in runs[1][k]
                  and not math.isnan(runs[0][k][c]) and not math.isnan(runs[1][k][c])]
        dropped = len(shared) - len(usable)
        a = [runs[0][k][c] for k in usable]
        b = [runs[1][k][c] for k in usable]
        if len(a) < 2:
            print(f"{CLASS_NAMES.get(c, c):<10}  only {len(a)} usable cases, skipped")
            continue
        ma, mb = st.mean(a), st.mean(b)
        diffs = [x - y for x, y in zip(a, b)]
        sd = st.stdev(diffs) if len(diffs) > 1 else 0.0
        p = wilcoxon_signed_rank(diffs)
        spreads.append(abs(ma - mb))
        report[c] = {"name": CLASS_NAMES.get(c, c), "seed1": ma, "seed2": mb,
                     "spread": abs(ma - mb), "per_case_sd": sd, "paired_p": p,
                     "n_used": len(a), "n_dropped_absent_class": dropped}
        ps = f"{p:.3f}" if p is not None else "n/a"
        note = f"  ({len(a)} cases" + (f", {dropped} lack the class)" if dropped else ")")
        print(f"{CLASS_NAMES.get(c, c):<10}{ma:>10.4f}{mb:>10.4f}"
              f"{abs(ma-mb):>10.4f}{sd:>13.4f}{ps:>10}{note}")

    worst = max(spreads) if spreads else 0.0
    mean_spread = st.mean(spreads) if spreads else 0.0
    print(f"\nmean spread {mean_spread:.4f}, worst class {worst:.4f}")
    print(f"effect being chased {TARGET_EFFECT:.4f} "
          f"(MedNeXt's kernel 5 gain over kernel 3, 0.2 Dice points)")

    if worst < TARGET_EFFECT / 2:
        verdict = "GO"
        note = "spread is well under the effect; one run per placement is interpretable"
    elif worst < TARGET_EFFECT * 1.5:
        verdict = "MARGINAL"
        note = ("spread is comparable to the effect; placements need repeats, "
                "not more configurations")
    else:
        verdict = "NO GO"
        note = ("spread exceeds the effect. The accuracy axis is not measurable "
                "at this budget on this task. Report the cost-side dominance "
                "result, which requires no training, and state plainly why the "
                "accuracy comparison is not attempted")
    print(f"\nVERDICT: {verdict}\n  {note}")

    if verdict != "GO" and spreads:
        tight = min(spreads)
        print(f"\n  The tightest class has spread {tight:.4f}. Resolving "
              f"{TARGET_EFFECT:.4f} at that noise level needs roughly "
              f"{max(1, round((tight / TARGET_EFFECT) ** 2))} runs per placement.")

    json.dump({"summaries": [str(p) for p in paths], "classes": report,
               "mean_spread": mean_spread, "worst_spread": worst,
               "target_effect": TARGET_EFFECT, "verdict": verdict},
              open(ROOT / "results" / "variance_pilot.json", "w"), indent=1)
    print(f"\nwrote {ROOT / 'results' / 'variance_pilot.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
