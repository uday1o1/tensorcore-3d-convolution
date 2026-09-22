"""How much of a measured architectural difference is just the seed?

Takes two or more training runs of one identical configuration, differing only
in random seed, and reports every pairwise comparison between them. The spread
is the noise floor of this training setup: any claim that one architecture beats
another has to clear it.

The threshold is not arbitrary. MedNeXt reports uniform kernel 5 with UpKern
improving mean Dice by roughly 0.2 points over kernel 3 (84.03 to 84.23 on BTCV,
86.71 to 87.06 on AMOS22). That is the size of the effect a placement study
chases.

WHY EVERY PAIR, NOT JUST ONE. With two runs there is a single comparison, which
can show that a false positive is possible but not how often it occurs, and an
unlucky draw is the obvious objection. Three runs give three comparisons and a
rate.

WHICH NUMBER TO COMPARE. nnU-Net logs an "Average global foreground Dice" every
epoch, computed on training-time validation batches. It is extremely noisy: in
one run consecutive epochs gave tumour Dice of 0.79, 0.63, 0.74, 0.79, 0.50,
0.84. That is a training diagnostic, not a result, and an earlier version of
this script compared it, which would have declared the study impossible on the
strength of the wrong quantity. The result is the per-case Dice in
validation_raw/summary.json, written after training by full sliding window
inference over every validation case.

Per-case values also permit a paired signed-rank test over cases, which is the
test this project's proposal specified, and which is the point: run against a
known null, it should not find anything.

Usage:
  python analyze_variance_pilot.py <summary1.json> <summary2.json> [more...]
  python analyze_variance_pilot.py          # all pilot seeds present on disk
"""
import itertools
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# MedNeXt's reported gain for uniform kernel 5 over kernel 3, on a 0-1 scale.
TARGET_EFFECT = 0.002
ALPHA = 0.05

RESULTS = Path("/workspace/RESULTS_FOLDER/nnUNet/3d_fullres/Task003_Liver")
TRAINER = "nnUNetTrainerV2_MedNeXt_B_k3_150ep_s{}__nnUNetPlansv2.1_trgSp_1x1x1"
CLASS_NAMES = {"1": "liver", "2": "tumor"}


def default_paths():
    out = []
    for seed in (1, 2, 3, 4, 5):
        p = RESULTS / TRAINER.format(seed) / "fold_0" / "validation_raw" / "summary.json"
        if p.exists():
            out.append(p)
    return out


def per_case(path):
    """Per-case Dice by class label, keyed by case identifier."""
    d = json.load(open(path))
    out = {}
    for case in d["results"]["all"]:
        ref = case.get("reference") or case.get("test") or ""
        out[Path(str(ref)).name] = {
            k: v["Dice"] for k, v in case.items()
            if k.isdigit() and isinstance(v, dict) and "Dice" in v}
    return out


def wilcoxon_signed_rank(diffs):
    """Two-sided Wilcoxon signed-rank p, normal approximation.

    Returns None when too few nonzero differences for the approximation to mean
    anything, rather than a number that would look like evidence.
    """
    nz = [d for d in diffs if d != 0 and not math.isnan(d)]
    n = len(nz)
    if n < 6:
        return None
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    for pos, i in enumerate(order, start=1):
        ranks[i] = float(pos)
    w_plus = sum(ranks[i] for i in range(n) if nz[i] > 0)
    mean = n * (n + 1) / 4
    sd = (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    if sd == 0:
        return None
    z = (w_plus - mean) / sd
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def compare(run_a, run_b, cls):
    """Paired comparison on one class, dropping cases where it is absent.

    A class Dice is NaN for cases that do not contain that class, and those
    cases carry no information about it. Coercing them to zero would drag both
    means down and shrink the apparent spread, making the noise floor look
    smaller than it is.
    """
    shared = sorted(set(run_a) & set(run_b))
    usable = [k for k in shared
              if cls in run_a[k] and cls in run_b[k]
              and not math.isnan(run_a[k][cls]) and not math.isnan(run_b[k][cls])]
    if len(usable) < 2:
        return None
    a = [run_a[k][cls] for k in usable]
    b = [run_b[k][cls] for k in usable]
    diffs = [x - y for x, y in zip(a, b)]
    return {"n": len(usable), "dropped": len(shared) - len(usable),
            "mean_a": st.mean(a), "mean_b": st.mean(b),
            "spread": abs(st.mean(a) - st.mean(b)),
            "sd": st.stdev(diffs) if len(diffs) > 1 else 0.0,
            "p": wilcoxon_signed_rank(diffs)}


def main(argv):
    paths = [Path(p) for p in argv[1:]] or default_paths()
    if len(paths) < 2:
        print(f"need at least two run summaries, found {len(paths)}")
        for p in paths:
            print(f"  {p}")
        print("\nTraining AND validation must finish for each seed. The")
        print("per-epoch Dice in the training log is a diagnostic, not a result.")
        return 1
    missing = [p for p in paths if not p.exists()]
    if missing:
        for m in missing:
            print(f"missing: {m}")
        return 1

    runs = [per_case(p) for p in paths]
    labels = []
    for i, p in enumerate(paths):
        parent = p.parts[-4]
        labels.append(parent.split("_s")[-1].split("__")[0] if "_s" in parent
                      else str(i + 1))
    classes = sorted({c for r in runs for v in r.values() for c in v} - {"0"})
    pairs = list(itertools.combinations(range(len(runs)), 2))

    print(f"{len(runs)} runs of one identical configuration, "
          f"{len(pairs)} pairwise comparisons")
    print(f"effect being chased {TARGET_EFFECT:.4f} "
          f"(0.2 Dice points, MedNeXt's kernel 5 gain over kernel 3)\n")

    report = {}
    for cls in classes:
        name = CLASS_NAMES.get(cls, cls)
        print(f"{name}")
        print(f"  {'pair':<10}{'mean A':>9}{'mean B':>9}{'spread':>9}"
              f"{'sd':>9}{'paired p':>10}{'n':>5}")
        rows = []
        for i, j in pairs:
            r = compare(runs[i], runs[j], cls)
            if r is None:
                continue
            r["pair"] = f"s{labels[i]}/s{labels[j]}"
            rows.append(r)
            ps = f"{r['p']:.3f}" if r["p"] is not None else "n/a"
            flag = "  <- significant" if (r["p"] is not None and r["p"] < ALPHA) else ""
            print(f"  {r['pair']:<10}{r['mean_a']:>9.4f}{r['mean_b']:>9.4f}"
                  f"{r['spread']:>9.4f}{r['sd']:>9.4f}{ps:>10}{r['n']:>5}{flag}")
        if not rows:
            continue
        spreads = [r["spread"] for r in rows]
        sig = [r for r in rows if r["p"] is not None and r["p"] < ALPHA]
        worst = max(spreads)
        print(f"  spread: min {min(spreads):.4f}, max {worst:.4f}, "
              f"vs effect {TARGET_EFFECT:.4f} ({worst/TARGET_EFFECT:.0f}x)")
        print(f"  false positives: {len(sig)}/{len(rows)} comparisons reach "
              f"p < {ALPHA} between IDENTICAL configurations")
        if worst > 0:
            need = max(1, round((worst / TARGET_EFFECT) ** 2))
            print(f"  runs per configuration needed to resolve the effect: ~{need}")
        report[cls] = {"name": name, "pairs": rows, "worst_spread": worst,
                       "min_spread": min(spreads), "n_significant": len(sig),
                       "n_comparisons": len(rows)}
        print()

    worst_overall = max((v["worst_spread"] for v in report.values()), default=0.0)
    total_sig = sum(v["n_significant"] for v in report.values())
    total_cmp = sum(v["n_comparisons"] for v in report.values())

    if worst_overall < TARGET_EFFECT / 2:
        verdict, note = "GO", "spread is well under the effect; single runs are interpretable"
    elif worst_overall < TARGET_EFFECT * 1.5:
        verdict, note = "MARGINAL", "spread is comparable to the effect; repeats are required"
    else:
        verdict, note = "NO GO", ("spread exceeds the effect. The accuracy axis is not "
                                  "measurable at this budget, and the cost-side result "
                                  "stands on its own instead")
    print(f"VERDICT: {verdict}\n  {note}")
    print(f"  across all classes, {total_sig} of {total_cmp} paired tests between "
          f"identical configurations reached p < {ALPHA}")

    json.dump({"summaries": [str(p) for p in paths], "n_runs": len(runs),
               "classes": report, "worst_spread": worst_overall,
               "target_effect": TARGET_EFFECT, "verdict": verdict,
               "n_significant": total_sig, "n_comparisons": total_cmp},
              open(ROOT / "results" / "variance_pilot.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
