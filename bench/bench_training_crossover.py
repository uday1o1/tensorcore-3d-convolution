"""Does the crossover move when you train rather than infer?

Every measurement in this project so far, and every measurement in the Im2win
papers, is forward only. The cost that actually constrains large kernel 3D
architectures is training cost, and there is a structural reason to expect the
two to differ.

The windowed method loses because it materializes a buffer that implicit GEMM
never builds. In the forward pass that buffer is built once and consumed once:

    out       = W  @ filters

In the backward pass the weight gradient contracts the SAME materialized
buffer against the output gradient:

    dfilters  = W^T @ dout

So across a training step the materialization is paid once and consumed twice.
Under the arithmetic intensity account that governs every other result here,
the quantity deciding the trade is compute per materialized byte, and that
quantity roughly doubles when the buffer serves both passes.

PREDICTION, recorded before measuring: the windowed method should do better
relative to cuDNN in training than in inference, and the crossover should move
to SMALLER kernel sizes. Configurations that lose on forward-only timing but
sit close to parity should cross over once the backward pass is included.

This is falsifiable in the other direction too. The windowed path's backward
also has to propagate gradients through the unfold, which is a scatter-add
over overlapping windows and has no counterpart in cuDNN's backward. If that
scatter costs more than the reuse saves, the crossover moves the wrong way and
the prediction is refuted.

Reports, per configuration, the cuDNN-to-windowed ratio for forward alone and
for a full forward-plus-backward step, so the shift is read directly.

Usage:  python bench_training_crossover.py [trials]
"""
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Resolve paths from this file, not the working directory, so the script
# runs from anywhere in a fresh clone.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from winconv import im2win_conv3d_minmat

torch.backends.cudnn.benchmark = True

CHANNELS = [30, 60, 120, 240]
SPATIAL = [16, 32, 64]
KERNELS = [3, 5, 7, 9, 11]
BATCH = 2



def device_tag():
    """Short filesystem-safe name for the GPU, so multi-device runs coexist."""
    name = torch.cuda.get_device_name(0)
    return (name.replace("NVIDIA ", "").replace("GeForce ", "")
                .replace(" ", "").replace("/", "-").lower())


def device_meta():
    """Everything needed to interpret a timing later."""
    p = torch.cuda.get_device_properties(0)
    return {"gpu": torch.cuda.get_device_name(0),
            "capability": f"{p.major}.{p.minor}",
            "sm_count": p.multi_processor_count,
            "total_mem_gb": round(p.total_memory / 1e9, 1),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version()}


def timeit(fn, n=6, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def main(trials=5):
    rows = []
    print(f'{"C":>5}{"sp":>4}{"k":>3} | {"fwd only":>18} | {"fwd+bwd":>18} | shift')
    print(f'{"":>5}{"":>4}{"":>3} | {"cuDNN":>8}{"ratio":>10} | {"cuDNN":>8}{"ratio":>10} |')
    for C in CHANNELS:
        for sp in SPATIAL:
            for k in KERNELS:
                if sp - k + 1 < 4:
                    continue
                try:
                    x = torch.randn(BATCH, C, sp, sp, sp, device="cuda",
                                    dtype=torch.half, requires_grad=True)
                    w = torch.randn(C, C, k, k, k, device="cuda",
                                    dtype=torch.half, requires_grad=True)

                    def fwd_ref():
                        with torch.no_grad():
                            F.conv3d(x, w)

                    def fwd_win():
                        with torch.no_grad():
                            im2win_conv3d_minmat(x, w)

                    def step(fn):
                        def go():
                            if x.grad is not None:
                                x.grad = None
                            if w.grad is not None:
                                w.grad = None
                            out = fn()
                            out.sum().backward()
                        return go

                    step_ref = step(lambda: F.conv3d(x, w))
                    step_win = step(lambda: im2win_conv3d_minmat(x, w))

                    f_ref = st.median([timeit(fwd_ref) for _ in range(trials)])
                    f_win = st.median([timeit(fwd_win) for _ in range(trials)])
                    t_ref = st.median([timeit(step_ref) for _ in range(trials)])
                    t_win = st.median([timeit(step_win) for _ in range(trials)])

                    # The reuse that helps latency has a memory price: the
                    # materialized buffer must be RETAINED for the backward
                    # pass, where in inference it is freed immediately. Measure
                    # it, since memory is half of what makes large kernel 3D
                    # training prohibitive in the first place.
                    mem = {}
                    for name, go in (("cudnn", step_ref), ("win", step_win)):
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        go()
                        torch.cuda.synchronize()
                        mem[name] = torch.cuda.max_memory_allocated() / 1e9

                    rf, rt = f_ref / f_win, t_ref / t_win
                    rows.append({**device_meta(), "C": C, "sp": sp, "k": k, "batch": BATCH,
                                 "trials": trials,
                                 "fwd_cudnn_ms": f_ref * 1000,
                                 "fwd_win_ms": f_win * 1000, "fwd_ratio": rf,
                                 "step_cudnn_ms": t_ref * 1000,
                                 "step_win_ms": t_win * 1000, "step_ratio": rt,
                                 "shift": rt - rf,
                                 "step_mem_cudnn_gb": mem["cudnn"],
                                 "step_mem_win_gb": mem["win"],
                                 "mem_ratio": mem["win"] / max(mem["cudnn"], 1e-9)})
                    print(f"{C:5d}{sp:4d}{k:3d} | {f_ref*1000:8.2f}{rf:10.2f} | "
                          f"{t_ref*1000:8.2f}{rt:10.2f} | {rt-rf:+6.2f} | "
                          f"mem x{mem['win']/max(mem['cudnn'],1e-9):5.2f}", flush=True)
                    del x, w
                    torch.cuda.empty_cache()
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    torch.cuda.empty_cache()
                    print(f"{C:5d}{sp:4d}{k:3d} |  skipped ({type(e).__name__})", flush=True)

    if not rows:
        print("nothing measured")
        return 1
    json.dump(rows, open(ROOT / "results" / f"training_crossover_{device_tag()}.json", "w"), indent=1)

    shifts = [r["shift"] for r in rows]
    better = sum(s > 0 for s in shifts)
    fwd_wins = sum(r["fwd_ratio"] > 1 for r in rows)
    step_wins = sum(r["step_ratio"] > 1 for r in rows)
    geo_f = math.exp(sum(math.log(r["fwd_ratio"]) for r in rows) / len(rows))
    geo_t = math.exp(sum(math.log(r["step_ratio"]) for r in rows) / len(rows))

    print(f"\nconfigurations measured: {len(rows)}")
    print(f"  windowed wins on forward only    : {fwd_wins}")
    print(f"  windowed wins on a training step : {step_wins}")
    print(f"  geometric mean ratio, forward    : {geo_f:.3f}")
    print(f"  geometric mean ratio, step       : {geo_t:.3f}")
    print(f"  configurations improving in training: {better}/{len(rows)}")
    print(f"  median shift: {st.median(shifts):+.3f}")

    verdict = "CONFIRMED" if geo_t > geo_f else "REFUTED"
    print(f"\nprediction 'the crossover moves earlier in training': {verdict}")
    mr = [r["mem_ratio"] for r in rows]
    print(f"  peak training memory, windowed over cuDNN: median {st.median(mr):.2f}x, "
          f"worst {max(mr):.2f}x")
    print("  (the buffer that is reused across both passes must also be retained "
          "across them)")

    # Back out the backward pass alone. This is the quantity no published
    # comparison in this family reports, since they all time forward kernels.
    #
    # Caveat stated in the output: the forward baseline here is measured under
    # no_grad, so subtracting it from the full step slightly overstates the
    # backward by attributing graph construction and activation saving to it.
    # The direction of that bias is the same for both algorithms.
    bwd = []
    for r in rows:
        bc = r["step_cudnn_ms"] - r["fwd_cudnn_ms"]
        bw = r["step_win_ms"] - r["fwd_win_ms"]
        if bc > 0 and bw > 0:
            bwd.append({"C": r["C"], "sp": r["sp"], "k": r["k"],
                        "bwd_cudnn_ms": bc, "bwd_win_ms": bw, "bwd_ratio": bc / bw,
                        "fwd_ratio": r["fwd_ratio"]})
    if bwd:
        br = [b["bwd_ratio"] for b in bwd]
        fr = [b["fwd_ratio"] for b in bwd]
        geo_b = math.exp(sum(math.log(v) for v in br) / len(br))
        print(f"\nbackward pass alone, inferred as step minus forward "
              f"({len(bwd)} configurations):")
        print(f"  geometric mean ratio, backward   : {geo_b:.3f}")
        print(f"  spread of forward ratios         : {min(fr):.2f} to {max(fr):.2f}")
        print(f"  spread of backward ratios        : {min(br):.2f} to {max(br):.2f}")
        print(f"  windowed wins on backward        : {sum(v > 1 for v in br)}/{len(br)}")
        print("  (forward baseline is no_grad, so this slightly overstates the "
              "backward for both algorithms)")

    # Where does the crossover sit in each regime?
    for label, key in (("forward only", "fwd_ratio"), ("training step", "step_ratio")):
        firsts = []
        for C in CHANNELS:
            for sp in SPATIAL:
                sel = sorted((r for r in rows if r["C"] == C and r["sp"] == sp),
                             key=lambda r: r["k"])
                hit = next((r["k"] for r in sel if r[key] > 1.0), None)
                if hit:
                    firsts.append(hit)
        if firsts:
            print(f"  smallest winning kernel, {label}: median {st.median(firsts):.0f} "
                  f"over {len(firsts)} shapes that cross at all")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 5))
