"""Does placing large kernels deep buy accuracy, and does it need UpKern?

Reads the per-case validation summaries in results/validation/ and compares
every trained configuration against the MedNeXt-B kernel 3 baseline, on the
same fold, the same plans and the same evaluation path.

Three comparison families:
  architecture       3-3-5-5, 3-5-5-7, 5-5-5-5, all UpKern initialized
  training strategy  3-5-5-7 UpKern against 3-5-5-7 from scratch
  control            each UpKern placement against 3-3-3-3 continued from the
                     same source by the same protocol, which separates what
                     the kernels add from what UpKern's extra training adds

A difference is only read as one if it exceeds the seed spread measured by the
variance pilot (results/variance_pilot.json) for that class. The pilot showed
that a paired signed-rank test across cases calls two identical networks
different, so the p value is reported but never decides anything on its own.

Which cases count. A class is scored only on cases whose reference contains
it. nnU-Net's own mean does otherwise: on a case without tumour it gives Dice 0
if the network predicts any tumour voxel and leaves the case out if it predicts
none, so two runs are averaged over different sets of cases and a few stray
voxels on one tumour-free scan move the mean by as much as a real difference.
In this study one run marked 36 voxels on a tumour-free case and was averaged
over 26 cases against the baseline's 25. False detections on cases without the
class are reported separately, as a count of cases and voxels, and nnU-Net's
own mean is kept alongside for comparison.

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
from analyze_variance_pilot import wilcoxon_signed_rank, CLASS_NAMES

VALIDATION = ROOT / "results" / "validation"
BASELINE = "nnUNetTrainerV2_MedNeXt_B_k3_150ep_s1"
# trainer -> (placement, initialization)
RUNS = {
    BASELINE: ("3-3-3-3", "trained"),
    "nnUNetTrainerV2_MedNeXt_B_place_3_3_5_5_150ep": ("3-3-5-5", "UpKern"),
    "nnUNetTrainerV2_MedNeXt_B_place_3_5_5_7_150ep": ("3-5-5-7", "UpKern"),
    "nnUNetTrainerV2_MedNeXt_B_place_5_5_5_5_150ep": ("5-5-5-5", "UpKern"),
    "nnUNetTrainerV2_MedNeXt_B_place_3_5_5_7_scratch_150ep": ("3-5-5-7", "scratch"),
    "nnUNetTrainerV2_MedNeXt_B_place_3_3_3_3_continued_150ep": ("3-3-3-3", "continued"),
    "nnUNetTrainerV2_150ep": ("nnU-Net", "trained"),
}
CONTROL = "nnUNetTrainerV2_MedNeXt_B_place_3_3_3_3_continued_150ep"


def noise_floor(n_present):
    """Seed spread per class from the variance pilot, on this script's case rule.

    The pilot averaged over cases scored in both runs (n_used), which for
    tumour included tumour-free cases where both runs scored 0. Every case that
    contains the class is scored in every run, so the pilot's set is those
    cases plus tumour-free ones contributing a difference of exactly zero. The
    summed difference is therefore the same on both rules and only the divisor
    changes, which converts the spread exactly without the pilot's raw files.
    """
    d = json.load(open(ROOT / "results" / "variance_pilot.json"))
    out = {}
    for cls, v in d["classes"].items():
        spread = v["spread"] if "spread" in v else v["worst_spread"]
        n_used = v.get("n_used", n_present.get(cls))
        out[cls] = spread * n_used / n_present[cls] if n_present.get(cls) else spread
    return out


def load(path):
    """Per case: {class: (dice, reference voxels, predicted voxels)}, and nnU-Net's mean."""
    d = json.load(open(path))
    cases = {}
    for case in d["results"]["all"]:
        cases[Path(str(case["reference"])).name] = {
            k: (v["Dice"], v["Total Positives Reference"], v["Total Positives Test"])
            for k, v in case.items() if k.isdigit() and isinstance(v, dict)}
    nnunet_mean = {k: v["Dice"] for k, v in d["results"]["mean"].items() if k.isdigit()}
    return cases, nnunet_mean


def present(run, cls):
    """Cases whose reference contains the class; Dice is defined on all of them."""
    return {k: v[cls][0] for k, v in run.items() if cls in v and v[cls][1] > 0}


def false_detections(run, cls):
    """Cases without the class in the reference where the network predicted it."""
    hits = [v[cls][2] for v in run.values() if cls in v and v[cls][1] == 0 and v[cls][2] > 0]
    return {"cases": len(hits), "voxels": sum(hits)}


def mean_dice(run, cls):
    vals = list(present(run, cls).values())
    return st.mean(vals) if vals else None


def paired(run, base, cls):
    a, b = present(run, cls), present(base, cls)
    keys = sorted(set(a) & set(b))
    diffs = [a[k] - b[k] for k in keys]
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
    loaded = {t: load(p) for t, p in found.items()}
    runs = {t: v[0] for t, v in loaded.items()}
    base = runs[BASELINE]
    floor = noise_floor({c: len(present(base, c)) for c in ("1", "2")})
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
            row[f"nnunet_mean_{CLASS_NAMES[cls]}"] = loaded[t][1].get(cls)
            row[f"false_detections_{CLASS_NAMES[cls]}"] = false_detections(runs[t], cls)
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
    print("Dice over cases containing the class; false detections on cases without it:")
    for row in rows:
        fd = row["false_detections_tumor"]
        label = row["placement"] + ("" if row["init"] in ("trained", "UpKern") else f" {row['init']}")
        print(f"  {label:<20}"
              f"tumour on {fd['cases']} tumour-free case(s), {fd['voxels']} voxels;"
              f"  nnU-Net's own means {row['nnunet_mean_liver']:.4f} / {row['nnunet_mean_tumor']:.4f}")

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

    # Kernels or extra training: UpKern placements against the same protocol
    # with nothing enlarged.
    control = None
    if CONTROL in runs:
        control = {}
        print("\nUpKern placements against 3-3-3-3 continued (same protocol, no larger kernels)")
        for t, (placement, init) in RUNS.items():
            if init != "UpKern" or t not in runs:
                continue
            control[placement] = {}
            for cls in classes:
                r = paired(runs[t], runs[CONTROL], cls)
                beyond = r["delta"] is not None and abs(r["delta"]) > floor[cls]
                control[placement][CLASS_NAMES[cls]] = {**r, "beyond_seed_noise": beyond}
                ps = f"{r['p']:.3f}" if r["p"] is not None else "n/a"
                print(f"  {placement:<8}{CLASS_NAMES[cls]:<7}{r['delta']:+.4f}  p {ps}  "
                      f"{'beyond' if beyond else 'within'} seed noise {floor[cls]:.4f}")

    json.dump({"baseline": BASELINE, "noise_floor": {CLASS_NAMES[c]: floor[c] for c in classes},
               "rows": rows, "upkern_vs_scratch": strategy,
               "upkern_vs_continued": control},
              open(results_dir() / "training_study.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
