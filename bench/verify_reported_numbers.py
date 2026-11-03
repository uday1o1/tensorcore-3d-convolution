"""Recomputes the derived numbers the write-up quotes from committed data in
results/ (the CLAIMED table).

Most reported numbers come from the RTX 3090, the GPU class Im2win's own
evaluation used. The device and host comparisons also read the RTX 4090 files
and results/host2/, and the trained placement runs span both hosts
(results/validation/training_host_*.txt).

The measured data in results/ is raw: per-configuration timings. Most of the
claims made about it are derived, things like win counts, medians, geometric
means against an oracle, and rule agreement rates. A reader has no way to check
those by eye, and a transcription error between the data and the prose would be
invisible.

This script recomputes the derived numbers listed in CLAIMED from the committed JSON and prints them
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
CLAIMED = {
    "dense configurations won": 26,
    "depthwise configurations won": 2,
    "dense median ratio": "0.95",
    "depthwise median ratio": "0.83",
    "best dense ratio": "2.36",
    "best depthwise ratio": "1.03",
    "wins at k>=9": 19,
    "wins at C>=240 and k>=5": 11,
    "largest single-layer saving, ms": "735",
    "geo gap, always cuDNN": "1.132",
    "geo gap, always windowed": "1.160",
    "geo gap, shape rule": "1.021",
    "held out, geo gap always cuDNN": "1.126",
    "held out, geo gap always windowed": "1.098",
    "held out, geo gap shape rule": "1.041",
    "held out, rule agreements out of 32": 24,
    "training, geo ratio forward": "0.940",
    "training, geo ratio backward": "1.030",   # rtx3090 only; 0.937 on rtx4090
    "training, geo ratio full step": "1.015",
    "training, peak memory median": "2.16",
    # Part II, kernel placement
    "placement, 5-5-5-5 train cost": "2.00",
    "placement, 3-5-5-7 train cost": "1.32",
    "placement, 3-3-5-5 train cost": "1.08",
    "placement, 5-5-5-5 FLOP cost": "1.21",
    "pilot, tumor spread": "0.0407",
    "pilot, liver spread": "0.0005",
    "pilot, tumor paired p": "0.049",
    # Measurement audit: two sessions per GPU, the two GPUs on different hosts
    "audit, 3090 session winner changes": 4,
    "audit, 4090 session winner changes": 2,
    "audit, 3090 session median offset": "0.990",
    "audit, 4090 session median offset": "0.998",
    "audit, cross-host winner changes, one session": 17,
    "audit, cross-host winner changes, session-confirmed": 13,
    "audit, 3090 rule 2 confirmed by both sessions": 10,
    # Trained placement study (results/training_study.json)
    "study, kernel 3 liver Dice": "0.9642",
    "study, 3-3-5-5 liver Dice": "0.9652",
    "study, 3-5-5-7 liver Dice": "0.9653",
    "study, 5-5-5-5 liver Dice": "0.9653",
    "study, 3-5-5-7 scratch liver Dice": "0.9644",
    "study, kernel 3 continued liver Dice": "0.9650",
    "study, nnU-Net liver Dice": "0.9546",
    "study, 3-5-5-7 scratch liver gain": "0.0002",
    "study, continued liver gain": "0.0008",
    "study, nnU-Net liver gap": "-0.0095",
    "study, largest placement-vs-continued liver gap": "0.0003",
    "study, tumour seed spread on present cases": "0.0424",
    "study, cases containing tumour": 24,
    # Second RTX 4090 host (results/host2/)
    "hosts, second 4090 host session disagreements": 1,
    "hosts, 4090 host 1 vs host 2 winner changes": 0,
    "hosts, 3090 host vs 4090 host 2 winner changes": 14,
    "hosts, 4090 host 1 and host 2 settled configurations": 57,
    "audit, configurations settled on both the 3090 and 4090 hosts": 54,
    # The C>=240, k>=5 clause across devices
    "clause 2, 4090 grid wins out of 12": 8,
    "clause 2, 3090 held-out wins out of 6": 6,
    "clause 2, 4090 held-out wins out of 6": 3,
    "held out, rule margin over always cuDNN, 3090 percent": "8.25",
    "held out, rule margin over always cuDNN, 4090 percent": "0.01",
    # Part II table and prose
    "placement, 3-5-5-7 FLOP cost": "1.06",
    "placement, 3-5-5-7 receptive field": 371,
    "placement, 5-5-5-5 receptive field": 301,
    "placement, 3-5-5-7 90 percent mass": 73,
    "placement, 5-5-5-5 90 percent mass": 67,
    "study, 5-5-5-5 tumour gain over kernel 3": "+0.0465",
    "study, 5-5-5-5 tumour gain over continued": "+0.0056",
    "study, tumour comparisons beyond seed spread": 1,
}
# Integers are counts and must match exactly. Every other claim is written as
# a string with the decimals the prose prints, and must equal the computed
# value rounded to those decimals: a relative tolerance would pass a value the
# prose misrounds.


def decimals(claim):
    return len(claim.split(".")[1]) if "." in claim else 0


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
    rf_path = ROOT / "results" / "receptive_field_cost.json"
    pilot_path = ROOT / "results" / "variance_pilot.json"
    rf = json.load(open(rf_path)) if rf_path.exists() else []
    pilot = json.load(open(pilot_path)) if pilot_path.exists() else {}

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
        print("note: training_crossover absent, skipping training claims\n")

    if rf:
        by = {r["placement"]: r for r in rf}
        base = by["3-3-3-3"]
        for name in ("5-5-5-5", "3-5-5-7", "3-3-5-5"):
            if name in by:
                computed[f"placement, {name} train cost"] = (
                    by[name]["train_ms"] / base["train_ms"])
        if "5-5-5-5" in by:
            computed["placement, 5-5-5-5 FLOP cost"] = (
                by["5-5-5-5"]["gflops"] / base["gflops"])
    else:
        for k in list(CLAIMED):
            if k.startswith("placement,"):
                CLAIMED.pop(k)
        print("note: receptive_field_cost absent, skipping placement claims\n")

    if pilot:
        for c in pilot.get("classes", {}).values():
            name = c["name"]
            computed[f"pilot, {name} spread"] = c["worst_spread"]
            ps = [r["p"] for r in c["pairs"] if r.get("p") is not None]
            if ps:
                computed[f"pilot, {name} paired p"] = min(ps)
    else:
        for k in list(CLAIMED):
            if k.startswith("pilot,"):
                CLAIMED.pop(k)
        print("note: variance_pilot absent, skipping pilot claims\n")

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from compare_devices import load, session_agreement, same_gpu_two_hosts
        ag = session_agreement(load("crossover_map"), load("training_crossover"))
    except Exception as e:
        ag = {}
        print(f"note: session agreement unavailable ({type(e).__name__})\n")
    if "rtx3090" in ag and "rtx4090" in ag:
        a3, a4 = ag["rtx3090"], ag["rtx4090"]
        cells = sorted(set(a3["cells"]) & set(a4["cells"]))
        computed["audit, 3090 session winner changes"] = len(a3["disagree"])
        computed["audit, 4090 session winner changes"] = len(a4["disagree"])
        computed["audit, 3090 session median offset"] = a3["median_offset"]
        computed["audit, 4090 session median offset"] = a4["median_offset"]
        computed["audit, cross-host winner changes, one session"] = sum(
            (a3["cm"][k] > 1) != (a4["cm"][k] > 1) for k in cells)
        computed["audit, cross-host winner changes, session-confirmed"] = sum(
            a3["verdicts"][k] is not None and a4["verdicts"][k] is not None
            and a3["verdicts"][k] != a4["verdicts"][k] for k in cells)
        computed["audit, configurations settled on both the 3090 and 4090 hosts"] = sum(
            a3["verdicts"][k] is not None and a4["verdicts"][k] is not None for k in cells)
        computed["audit, 3090 rule 2 confirmed by both sessions"] = sum(
            a3["cm"][k] > 1 and a3["tc"][k] > 1 for k in a3["cells"]
            if k[0] >= 240 and k[2] >= 5)
    else:
        for k in list(CLAIMED):
            if k.startswith("audit,"):
                CLAIMED.pop(k)
    h = same_gpu_two_hosts(ag) if ag else None
    if h:
        computed["hosts, second 4090 host session disagreements"] = len(h["host2_disagree"])
        computed["hosts, 4090 host 1 vs host 2 winner changes"] = len(h["same_gpu_changes"])
        computed["hosts, 3090 host vs 4090 host 2 winner changes"] = len(h["rtx3090_vs_host2_changes"])
        computed["hosts, 4090 host 1 and host 2 settled configurations"] = sum(
            ag["rtx4090"]["verdicts"][k] is not None and k not in h["host2_disagree"]
            for k in h["cells"])
    else:
        for k in list(CLAIMED):
            if k.startswith("hosts,"):
                CLAIMED.pop(k)

    cm4 = ROOT / "results" / "crossover_map_rtx4090.json"
    ho4 = ROOT / "results" / "dispatch_heldout_rtx4090.json"
    if cm4.exists() and ho4.exists():
        d4 = [r for r in json.load(open(cm4)) if r.get("mode", "dense") == "dense"]
        h4 = json.load(open(ho4))
        c2 = lambda r: r["C"] >= 240 and r["k"] >= 5
        computed["clause 2, 4090 grid wins out of 12"] = sum(r["ratio"] > 1 for r in d4 if c2(r))
        computed["clause 2, 3090 held-out wins out of 6"] = sum(
            r["windowed_actually_faster"] for r in held if c2(r))
        computed["clause 2, 4090 held-out wins out of 6"] = sum(
            r["windowed_actually_faster"] for r in h4 if c2(r))
        for tag, rows in (("3090", held), ("4090", h4)):
            computed[f"held out, rule margin over always cuDNN, {tag} percent"] = 100 * (
                geo_gap(rows, lambda r: False)
                / geo_gap(rows, lambda r: r["rule_says_windowed"]) - 1)
    else:
        for k in list(CLAIMED):
            if k.startswith("clause 2,") or "margin over" in k:
                CLAIMED.pop(k)

    if rf:
        computed["placement, 3-5-5-7 FLOP cost"] = by["3-5-5-7"]["gflops"] / base["gflops"]
    erf_path = ROOT / "results" / "effective_receptive_field.json"
    if erf_path.exists():
        erf = {r["placement"]: r for r in json.load(open(erf_path))}
        for n in ("3-5-5-7", "5-5-5-5"):
            computed[f"placement, {n} receptive field"] = erf[n]["theoretical"]
            computed[f"placement, {n} 90 percent mass"] = erf[n]["mass90"]

    study_path = ROOT / "results" / "training_study.json"
    if study_path.exists():
        sd = json.load(open(study_path))
        by = {r["trainer"]: r for r in sd["rows"]}
        k3 = by["nnUNetTrainerV2_MedNeXt_B_k3_150ep_s1"]["dice_liver"]
        name = {"nnUNetTrainerV2_MedNeXt_B_k3_150ep_s1": "kernel 3",
                "nnUNetTrainerV2_MedNeXt_B_place_3_3_5_5_150ep": "3-3-5-5",
                "nnUNetTrainerV2_MedNeXt_B_place_3_5_5_7_150ep": "3-5-5-7",
                "nnUNetTrainerV2_MedNeXt_B_place_5_5_5_5_150ep": "5-5-5-5",
                "nnUNetTrainerV2_MedNeXt_B_place_3_5_5_7_scratch_150ep": "3-5-5-7 scratch",
                "nnUNetTrainerV2_MedNeXt_B_place_3_3_3_3_continued_150ep": "kernel 3 continued",
                "nnUNetTrainerV2_150ep": "nnU-Net"}
        for t, n in name.items():
            computed[f"study, {n} liver Dice"] = by[t]["dice_liver"]
        computed["study, 3-5-5-7 scratch liver gain"] = by["nnUNetTrainerV2_MedNeXt_B_place_3_5_5_7_scratch_150ep"]["dice_liver"] - k3
        computed["study, continued liver gain"] = by["nnUNetTrainerV2_MedNeXt_B_place_3_3_3_3_continued_150ep"]["dice_liver"] - k3
        computed["study, nnU-Net liver gap"] = by["nnUNetTrainerV2_150ep"]["dice_liver"] - k3
        computed["study, largest placement-vs-continued liver gap"] = max(
            abs(v["liver"]["delta"]) for v in sd["upkern_vs_continued"].values())
        computed["study, tumour seed spread on present cases"] = sd["noise_floor"]["tumor"]
        computed["study, cases containing tumour"] = by["nnUNetTrainerV2_MedNeXt_B_place_3_5_5_7_150ep"]["tumor"]["n"]
        computed["study, 5-5-5-5 tumour gain over kernel 3"] = by["nnUNetTrainerV2_MedNeXt_B_place_5_5_5_5_150ep"]["tumor"]["delta"]
        computed["study, 5-5-5-5 tumour gain over continued"] = sd["upkern_vs_continued"]["5-5-5-5"]["tumor"]["delta"]
        computed["study, tumour comparisons beyond seed spread"] = (
            sum(r.get("tumor", {}).get("beyond_seed_noise", False) for r in sd["rows"])
            + sum(v["tumor"]["beyond_seed_noise"] for v in sd["upkern_vs_continued"].values())
            + sum([sd["upkern_vs_scratch"]["tumor"]["beyond_seed_noise"]]))
    else:
        for k in list(CLAIMED):
            if k.startswith("study,"):
                CLAIMED.pop(k)

    print(f"crossover grid: {len(dense)} dense, {len(depth)} depthwise, "
          f"{dense[0].get('trials','?')} trials each")
    print(f"held out grid:  {len(held)} configurations\n")
    print(f"  {'claim':<60}{'stated':>10}{'computed':>12}")
    ok = True
    for name, claim in CLAIMED.items():
        got = computed[name]
        if isinstance(claim, int):
            good = got == claim
        else:
            d = decimals(claim)
            good = f"{got:+.{d}f}" == f"{float(claim):+.{d}f}"
        ok &= good
        print(f"  {name:<60}{claim:>10}{got:>12.4f}  "
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
