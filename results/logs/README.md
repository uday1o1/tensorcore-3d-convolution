# Raw run logs

Verbatim stdout from the runs that produced the reported numbers, kept because
a log is the authentic record of what was executed and when.

| Log | Produced | Regenerate with |
|---|---|---|
| `crossover7.log` | `../crossover_map.json`, dense and depthwise at 7 trials | `python bench/bench_crossover.py 7` |
| `dispatch.log` | `../dispatch_heldout.json`, held-out rule test | `python bench/bench_dispatch.py 5` |
| `netdispatch.log` | end-to-end dispatcher at kernel sizes 3 and 5 | `python bench/bench_dispatch_network.py 3 5` |
| `netdispatch79.log` | end-to-end dispatcher at kernel sizes 7 and 9 | `python bench/bench_dispatch_network.py 7 9` |

The two `netdispatch` runs predate the change that made
`bench_dispatch_network.py` write `../dispatch_network.json`, so their numbers
live here rather than in a JSON file. Re-running either command now produces
that JSON.

| `cross4090.log` | `../crossover_map_rtx4090.json` | `python bench/bench_crossover.py 7` |
| `train4090.log` | `../training_crossover_rtx4090.json` | `python bench/bench_training_crossover.py 5` |
| `disp4090.log` | `../dispatch_heldout_rtx4090.json` | `python bench/bench_dispatch.py 5` |

Two devices, with PyTorch and cuDNN pinned to the same versions on both so
that cross-device differences are silicon rather than library version:

- RTX 3090, sm_86, driver 595.71.05
- RTX 4090, sm_89, 128 SMs, driver 570.144

Both on CUDA 12.8, PyTorch 2.11.0+cu128, Python 3.12, cuDNN 9.19.0.

The 4090's host shipped PyTorch 2.8.0 with cuDNN 9.10.2. Comparing against
that would have confounded hardware with library version, since cuDNN's
version determines which algorithms it can select, so the 4090 runs used an
isolated environment pinned to match the 3090 exactly.

`disp4090.log` was produced before the preflight was made device-aware, so its
preflight table validates against RTX 3090 reference values. That affects only
the gate, not the measurement: the held-out comparison times both algorithms
directly and does not use the autotuner.
