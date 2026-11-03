# Raw run logs

Verbatim console output (stdout and stderr) from the runs that produced the reported numbers, kept because
a log is the authentic record of what was executed and when. The Part I
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

`bench_dispatch_network.py` writes `../dispatch_network.json`; the two
`netdispatch` runs have no committed JSON, so their numbers live in these logs.
`disp4090.log` validates its preflight table against the RTX 3090 reference
values; that affects only the gate, not the measurement, since the held-out
comparison times both algorithms directly.

## Training

The `train_*.log` files are nnU-Net's own training logs
(`fold_0/training_log_*.txt` in each run directory), renamed; `pilot_s1.log`
and `pilot_s2.log` are the console output of the two pilot runs. Both carry
per-epoch losses, the online Dice estimate and epoch time. The scores reported for each run are the per-case summaries in
`../validation/` (for the two pilot runs, `../variance_pilot.json`), not the
online estimate here.

| Log | Run | Training host |
|---|---|---|
| `pilot_s1.log`, `pilot_s2.log` | MedNeXt-B kernel 3, seeds 1 and 2, the seed-spread pair behind `../variance_pilot.json` | not recorded in the logs |
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

PyTorch 2.11.0+cu128 (CUDA 12.8) with cuDNN 9.19.0 on every host, so that
cross-device differences are not library version; they remain host as well as
GPU differences.

- RTX 3090 benchmark host: driver 595.71.05
  (`measurements/host_rtx3090_main_versions.log`).
- RTX 4090 benchmark host: sm_89, 128 SMs; versions recorded in each row of the
  `rtx4090` JSON files.
- Second RTX 4090 host, also the RTX 4090 training host: driver 580.95.05, AMD
  EPYC 7742, 400 W power limit (`../host2/host.txt`).
- RTX 3090 training host: driver 580.126.09, AMD Ryzen 9 5950X, 320 W power
  limit (`../validation/training_host_rtx3090.txt`).
