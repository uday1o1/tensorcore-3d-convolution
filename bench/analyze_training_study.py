"""Does placing large kernels deep buy accuracy, and does it need UpKern?

Reads the per-case validation summaries in results/validation/ and compares
every trained configuration against the MedNeXt-B kernel 3 baseline, on the
same fold, the same plans and the same evaluation path.

Two questions, one per comparison family:
  architecture       3-3-5-5, 3-5-5-7, 5-5-5-5, all UpKern initialized
  training strategy  3-5-5-7 UpKern against 3-5-5-7 from scratch

A difference is only read as one if it exceeds the seed spread measured by the
variance pilot (results/variance_pilot.json) for that class. The pilot showed
that a paired signed-rank test across cases calls two identical networks
different, so the p value is reported but never decides anything on its own.

Cost and receptive field come from the committed cost measurements, so the
output is the Pareto table with an accuracy column, not a separate result.

Usage:
  python bench/analyze_training_study.py
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))
from benchenv import results_dir
from analyze_variance_pilot import per_case, wilcoxon_signed_rank, CLASS_NAMES

VALIDATION = ROOT / "results" / "validation"
BASELINE = "nnUNetTrainerV2_MedNeXt_B_k3_150ep_s1"
# trainer -> (placement, initialization)
RUNS = {
    BASELINE: ("3-3-3-3", "trained"),
    "nnUNetTrainerV2_MedNeXt_B_place_3_3_5_5_150ep": ("3-3-5-5", "UpKern"),
    "nnUNetTrainerV2_MedNeXt_B_place_3_5_5_7_150ep": ("3-5-5-7", "UpKern"),
    "nnUNetTrainerV2_MedNeXt_B_place_5_5_5_5_150ep": ("5-5-5-5", "UpKern"),
    "nnUNetTrainerV2_MedNeXt_B_place_3_5_5_7_scratch_150ep": ("3-5-5-7", "scratch"),
    "nnUNetTrainerV2_150ep": ("nnU-Net", "trained"),
}


def noise_floor():
    """Seed spread per class from the variance pilot, in either stored format."""
    d = json.load(open(ROOT / "results" / "variance_pilot.json"))
    out = {}
    for cls, v in d["classes"].items():
        out[cls] = v["spread"] if "spread" in v else v["worst_spread"]
    return out


def mean_dice(run, cls):
    vals = [v[cls] for v in run.values() if cls in v and not math.isnan(v[cls])]
    return st.mean(vals) if vals else None


def paired(run, base, cls):
    keys = [k for k in sorted(set(run) & set(base))
            if cls in run[k] and cls in base[k]
            and not math.isnan(run[k][cls]) and not math.isnan(base[k][cls])]
    diffs = [run[k][cls] - base[k][cls] for k in keys]
    return {"n": len(keys), "delta": st.mean(diffs) if diffs else None,
            "p": wilcoxon_signed_rank(diffs)}


def cost_table():
    cost = {r["placement"]: r for r in json.load(open(ROOT / "results" / "receptive_field_cost.json"))}
    erf = {r["placement"]: r for r in json.load(open(ROOT / "results" / "effective_receptive_field.json"))}
    base = cost["3-3-3-3"]["train_ms"]
    return {p: {"train_cost": cost[p]["train_ms"] / base,
                "rf_theoretical": erf[p]["theoretical"], "rf_mass90": erf[p]["mass90"]}
            for p in cost if p in erf}


def main():
    found = {t: VALIDATION / f"{t}_summary.json" for t in RUNS}
    found = {t: p for t, p in found.items() if p.exists()}
    if BASELINE not in found:
        print(f"baseline summary missing: {VALIDATION / (BASELINE + '_summary.json')}")
        return 1
    runs = {t: per_case(p) for t, p in found.items()}
    base = runs[BASELINE]
    floor = noise_floor()
    costs = cost_table()
    classes = ["1", "2"]

    print(f"{len(runs)} of {len(RUNS)} configurations evaluated; baseline {RUNS[BASELINE][0]}")
    print("seed noise floor: " + ", ".join(
        f"{CLASS_NAMES[c]} {floor[c]:.4f}" for c in classes) + "\n")
    hdr = f"{'config':<20}{'cost':>6}{'RF90':>6}"
    for c in classes:
        hdr += f"{CLASS_NAMES[c] + ' Dice':>12}{'delta':>9}{'p':>7}"
    print(hdr)

    rows = []
    for t, (placement, init) in RUNS.items():
        if t not in runs:
            continue
        name = placement if init in ("trained", "UpKern") else f"{placement} {init}"
        c = costs.get(placement, {})
        row = {"trainer": t, "placement": placement, "init": init,
               "n_cases": len(runs[t]), **c}
        line = (f"{name:<20}{c.get('train_cost', float('nan')):>6.2f}"
                f"{c.get('rf_mass90', float('nan')):>6}")
        for cls in classes:
            m = mean_dice(runs[t], cls)
            row[f"dice_{CLASS_NAMES[cls]}"] = m
            if t == BASELINE:
                line += f"{m:>12.4f}{'':>9}{'':>7}"
                continue
            r = paired(runs[t], base, cls)
            beyond = r["delta"] is not None and abs(r["delta"]) > floor[cls]
            row[CLASS_NAMES[cls]] = {**r, "beyond_seed_noise": beyond}
            ps = f"{r['p']:.3f}" if r["p"] is not None else "n/a"
            line += f"{m:>12.4f}{r['delta']:>+9.4f}{ps:>7}" + ("*" if beyond else " ")
        rows.append(row)
        print(line)
    print("\n* delta exceeds the seed spread for that class; nothing else is read as a difference")

    # Training strategy: the same architecture with and without UpKern.
    up = "nnUNetTrainerV2_MedNeXt_B_place_3_5_5_7_150ep"
    sc = "nnUNetTrainerV2_MedNeXt_B_place_3_5_5_7_scratch_150ep"
    strategy = None
    if up in runs and sc in runs:
        strategy = {}
        print("\n3-5-5-7, UpKern against scratch")
        for cls in classes:
            r = paired(runs[up], runs[sc], cls)
            beyond = r["delta"] is not None and abs(r["delta"]) > floor[cls]
            strategy[CLASS_NAMES[cls]] = {**r, "beyond_seed_noise": beyond}
            ps = f"{r['p']:.3f}" if r["p"] is not None else "n/a"
            print(f"  {CLASS_NAMES[cls]:<7} UpKern minus scratch {r['delta']:+.4f}  p {ps}  "
                  f"{'beyond' if beyond else 'within'} seed noise {floor[cls]:.4f}")

    json.dump({"baseline": BASELINE, "noise_floor": {CLASS_NAMES[c]: floor[c] for c in classes},
               "rows": rows, "upkern_vs_scratch": strategy},
              open(results_dir() / "training_study.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
