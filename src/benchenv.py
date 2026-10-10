"""Where benchmark output goes, and what is recorded about the machine.

Two rules this module exists to enforce.

Output location. Benchmarks write to results/ in the repository by default, so a
fresh clone reproduces the committed files in place. On a rented instance that
default breaks `git pull`: freshly written results are untracked or modified
files and the pull refuses to overwrite them. Setting BENCH_RESULTS_DIR sends
output outside the checkout, and the files are copied back deliberately.

Machine record. Timing results are only interpretable alongside the machine
that produced them, and hosts differ more than GPUs do. Comparing two sessions
of identical convolutions on one RTX 3090 host showed winners changing between
sessions, and our two-device comparison could not separate the GPU from the
host because neither the driver, the power limit, the clocks nor the host CPU
was recorded. Every timing row now carries them.

torch is imported lazily so results_dir() works on a machine without it.
"""
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def results_dir():
    """Directory for benchmark output: BENCH_RESULTS_DIR, else repo results/."""
    d = Path(os.environ.get("BENCH_RESULTS_DIR", ROOT / "results"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def device_tag():
    """Short filesystem-safe GPU name, so runs on different GPUs coexist."""
    import torch
    name = torch.cuda.get_device_name(0)
    return (name.replace("NVIDIA ", "").replace("GeForce ", "")
                .replace(" ", "").replace("/", "-").lower())


def _nvidia_smi(fields):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        vals = [v.strip() for v in out.split(",")]
        return dict(zip(fields, vals)) if len(vals) == len(fields) else {}
    except (OSError, subprocess.SubprocessError):
        return {}


def _cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def device_meta():
    """Everything needed to interpret a timing later, including the host.

    A power limit below the card's default, or a different driver, can move a
    ratio as much as a change of GPU generation, so both are recorded rather
    than assumed. Values nvidia-smi cannot report are left out, not guessed.
    """
    import torch
    p = torch.cuda.get_device_properties(0)
    meta = {"gpu": torch.cuda.get_device_name(0),
            "capability": f"{p.major}.{p.minor}",
            "sm_count": p.multi_processor_count,
            "total_mem_gb": round(p.total_memory / 1e9, 1),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "host_cpu": _cpu_model()}
    smi = _nvidia_smi(["driver_version", "power.limit", "power.default_limit",
                       "clocks.max.sm", "clocks.sm"])
    if smi:
        meta.update({"driver": smi.get("driver_version"),
                     "power_limit_w": smi.get("power.limit"),
                     "power_default_w": smi.get("power.default_limit"),
                     "sm_clock_max_mhz": smi.get("clocks.max.sm"),
                     "sm_clock_at_start_mhz": smi.get("clocks.sm")})
    return meta


def timed_pair(ref, win, timeit, trial):
    """Time both paths, alternating which goes first on successive trials.

    Every comparison in this project originally timed the baseline first and
    never reversed, so any position effect landed on one side of every ratio.
    Alternating the order cancels a position effect in the median and, because
    the order is recorded, also lets it be measured instead of assumed absent.

    Returns (t_ref, t_win, ref_first).
    """
    ref_first = trial % 2 == 0
    if ref_first:
        t_ref = timeit(ref)
        t_win = timeit(win)
    else:
        t_win = timeit(win)
        t_ref = timeit(ref)
    return t_ref, t_win, ref_first


def order_summary(pairs):
    """Median ratio for each order and their quotient, the position effect.

    pairs is a list of (t_ref, t_win, ref_first). A quotient near 1 means order
    did not matter; far from 1 means it did, and the order-balanced median is
    the number to report.
    """
    import statistics as st
    a = [r / w for r, w, f in pairs if f]
    b = [r / w for r, w, f in pairs if not f]
    if not a or not b:
        return {}
    ma, mb = st.median(a), st.median(b)
    return {"ratio_ref_first": ma, "ratio_win_first": mb,
            "order_effect": ma / mb if mb else None}


def discard_warmup(fn, n=3):
    """Run and throw away a few calls before the first timed measurement.

    The first measurement in a sequence is penalised: clocks ramp, cuDNN's
    autotuner searches, and allocators grow. Timing it counts setup as work.
    """
    import torch
    for _ in range(n):
        fn()
    torch.cuda.synchronize()


if __name__ == "__main__":
    # Self-check of the order logic, runnable without torch or a GPU.
    calls = []
    def fake_timeit(fn):
        calls.append(fn.__name__)
        return fn()
    def ref():
        return 2.0
    def win():
        return 1.0
    pairs = [timed_pair(ref, win, fake_timeit, t) for t in range(4)]
    assert calls == ["ref", "win", "win", "ref", "ref", "win", "win", "ref"], calls
    assert [f for _, _, f in pairs] == [True, False, True, False]
    s = order_summary(pairs)
    assert s["ratio_ref_first"] == s["ratio_win_first"] == 2.0 and s["order_effect"] == 1.0
    # A position effect that penalises whichever runs first must show up.
    biased = [(2.2, 1.0, True), (2.0, 1.1, False)] * 3
    assert order_summary(biased)["order_effect"] > 1.1
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        os.environ["BENCH_RESULTS_DIR"] = d
        assert results_dir() == Path(d)
        del os.environ["BENCH_RESULTS_DIR"]
    assert results_dir() == ROOT / "results"
    print("benchenv self-check passed")
