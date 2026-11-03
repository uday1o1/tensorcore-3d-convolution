# Raw run logs

Verbatim output from the runs that produced the reported numbers (console
stdout and stderr for the benchmarks and the pilot runs, nnU-Net's own training
logs for the trained runs). The Part I
measurements made with short inline scripts (the Im2win reproduction, the
generality audit, correctness, dimensional scaling, FFT, profiling and the
whole-network substitution) are in `measurements/`, indexed in its own README.

## Benchmarks

| Log | Produced | Regenerate with |
|---|---|---|
| `crossover7.log` | `../crossover_map_rtx3090.json`, dense and depthwise at 7 trials | `python bench/bench_crossover.py 7` |
| `dispatch.log` | `../dispatch_heldout_rtx3090.json`, held-out rule test | `python bench/bench_dispatch.py 5` |
| `traincross.log` | `../training_crossover_rtx3090.json`, forward against full training step | `python bench/bench_training_crossover.py 5` |
| `netdispatch.log` | end-to-end dispatcher at kernel sizes 3 and 5 | `python bench/bench_dispatch_network.py 3 5` |
| `netdispatch79.log` | end-to-end dispatcher at kernel sizes 7 and 9 | `python bench/bench_dispatch_network.py 7 9` |
| `cross4090.log` | `../crossover_map_rtx4090.json` | `python bench/bench_crossover.py 7` |
| `train4090.log` | `../training_crossover_rtx4090.json` | `python bench/bench_training_crossover.py 5` |
| `disp4090.log` | `../dispatch_heldout_rtx4090.json` | `python bench/bench_dispatch.py 5` |
| `crossover_host2_s1.log`, `crossover_host2_s2.log` | `../host2/crossover_map_s1.json`, `../host2/crossover_map_s2.json`, two sessions on the second RTX 4090 host | `python bench/bench_crossover.py 7`, run twice, renaming the output after each |
| `cudnn_conv9.log` | `../cudnn_conv9_runs_rtx3090.json`, 25 fresh processes on the second RTX 3090 host | `python bench/bench_cudnn_conv9.py 20` |
| `conv9_same_host.log` | the released Im2win conv9 kernel and cuDNN alternated five times on the second RTX 3090 host | `for i in 1 2 3 4 5; do echo "== round $i im2win"; ./sweep 256 64 56 56 64 3 1; echo "== round $i cudnn"; python bench/bench_cudnn_conv9.py 1 \| grep '^run'; done`, with `sweep` built from `bench/CMakeLists.txt` and the conv9 kernel source; each `bench_cudnn_conv9.py` call rewrites `cudnn_conv9_runs_<gpu>.json`, so run this before the `cudnn_conv9.log` sweep |
| `reproduce_im2win.log` | kernel-only and full-accounting ratios on conv9, second RTX 3090 host | `python bench/bench_reproduce_im2win.py` |
| `rf_controls.log` | `../receptive_field_cost_controls.json` | `python bench/bench_receptive_field.py --controls --tag controls` |
| `rf_p128.log` | `../receptive_field_cost_p128.json` | `python bench/bench_receptive_field.py --controls --patch 128 --tag p128` |
| `erf_controls.log` | `../effective_receptive_field_controls.json` | `python bench/bench_effective_receptive_field.py --controls` |
| `verify_correctness.log` | correctness of every implementation, second RTX 3090 host | `python bench/verify_correctness.py` |
| `setup_bench.log` | provisioning of the second RTX 3090 host, with its smoke test | `scripts/setup_vast.sh launch setup-bench` |

`bench_dispatch_network.py` writes `../dispatch_network.json`; the two
`netdispatch` runs have no JSON, so their numbers live in these logs.
`dispatch.log` and `disp4090.log` both check their preflight against fixed
RTX 3090 ratios (0.63, 1.45, 0.84, 1.28, 0.62, the same winners as
`../crossover_map_rtx3090.json`, which gives 0.65, 1.42, 0.84, 1.28, 0.65), not
against the device's own map; that affects only the gate, not the measurement,
since the held-out comparison times both algorithms directly.

The `rtx3090` and `rtx4090` crossover, held-out and training sweeps timed cuDNN
before the windowed path on every trial; the scripts in `bench/` alternate the
order, and the fixed-order versions are `git show 504759c^:bench/bench_crossover.py`
and likewise for `bench_dispatch.py` and `bench_training_crossover.py`.

## Training

The `train_*.log` files are nnU-Net's own training logs
(`fold_0/training_log_*.txt` in each run directory), renamed; `pilot_s1.log`
and `pilot_s2.log` are the console output of the two pilot runs. Both carry
per-epoch losses, the online Dice estimate and epoch time. The scores reported for each run are the per-case summaries in
`../validation/` (for the two pilot runs, `../variance_pilot.json`), not the
online estimate here.

| Log | Run | Training host |
|---|---|---|
| `pilot_s1.log`, `pilot_s2.log` | MedNeXt-B kernel 3, seeds 1 and 2, the seed-spread pair behind `../variance_pilot.json` | RTX 3090, driver 595.71.05 |
| `train_k3_s1.log` | MedNeXt-B kernel 3, the baseline and UpKern source | RTX 3090, `../validation/training_host_rtx3090.txt` |
| `train_place_3_5_5_7.log` | 3-5-5-7, UpKern from the kernel 3 run | RTX 3090, `../validation/training_host_rtx3090.txt` |
| `train_place_3_3_5_5.log` | 3-3-5-5, UpKern from the kernel 3 run | RTX 4090, `../validation/training_host_rtx4090.txt` |
| `train_place_5_5_5_5.log` | 5-5-5-5, UpKern from the kernel 3 run | RTX 4090, `../validation/training_host_rtx4090.txt` |
| `train_place_3_3_3_3_continued.log` | kernel 3 continued from the kernel 3 run, the UpKern control | RTX 4090, `../validation/training_host_rtx4090.txt` |
| `train_place_3_5_5_7_scratch.log` | 3-5-5-7, from scratch | RTX 4090, `../validation/training_host_rtx4090.txt` |
| `train_nnunet.log` | nnU-Net, same plans and fold | RTX 4090, `../validation/training_host_rtx4090.txt` |

Each run is regenerated with `scripts/setup_vast.sh train <Trainer>`, where the
trainer is the file of the same name in `src/trainers/`. The UpKern and
continued runs need the kernel 3 run's final checkpoint, so it trains first.

## Hosts

PyTorch 2.11.0+cu128 (CUDA 12.8) with cuDNN 9.19.0 on every host.

- RTX 3090, driver 595.71.05 (`measurements/host_rtx3090_main_versions.log`):
  the Part I measurement logs, the `rtx3090` sweeps and the variance pilot.
- RTX 4090, sm_89, 128 SMs: the `rtx4090` sweeps.
- Second RTX 4090, driver 580.95.05, AMD EPYC 7742, 400 W power limit
  (`../host2/host.txt`, `../validation/training_host_rtx4090.txt`): the two
  `host2` crossover sessions and the RTX 4090 training runs.
- RTX 3090 training host, driver 580.126.09, AMD Ryzen 9 5950X, 320 W power
  limit (`../validation/training_host_rtx3090.txt`).
- Second RTX 3090, driver 595.71.05, Intel Xeon E5-2673 v4, 320 W power limit
  (`host_rtx3090_second.txt`): the cuDNN conv9 sweep, the early-placement
  controls and the 128-cubed cost.
