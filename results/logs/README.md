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

Measured on NVIDIA RTX 3090 (24GB), sm_86, driver 595.71.05, CUDA 12.8,
PyTorch 2.11.0+cu128, Python 3.12.14, cuDNN 9.19.0.
