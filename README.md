# Convolution Cost in Volumetric Segmentation

**Author:** Uday Arora

## Abstract

Volumetric medical segmentation networks are built on 3D convolution, and its cost limits how far they can grow: MedNeXt notes that the computational requirements of indefinitely scaling kernel sizes in 3D networks quickly become prohibitive. This project asks two questions about that cost: can the convolution itself be made cheaper, and, since largely it cannot, where in a network is it worth spending?

Part I examines windowed convolution (Im2win), a memory-efficient method developed across papers from 2022 to 2026, all two-dimensional, whose most recent version reports 1.4 times cuDNN's overall Tensor Core throughput. Reproduced on the authors' 3x3 benchmark layer, the released kernel reaches 37.0 TFLOPS against cuDNN's 23.8 in dedicated timing runs, a ratio of 1.56, but the published timing leaves the window materialization outside the timed loop, and charging for it on the GPU moves the ratio to 1.13. cuDNN's own throughput on that layer varies between runs from 17.6 to 42.4 TFLOPS, so even the kernel-level comparison depends on which cuDNN run it is set against. The three released kernels whose launch grids we recorded use hardcoded grid dimensions, and the conv9 kernel, run at eight single-parameter variations of its benchmark shape, returns silently zero or partially computed output at seven of them. A shape-generic windowed convolution for 2D and 3D, verified against cuDNN, is 2.34 times slower on 3.9 times the memory inside a whole 3D segmentation network at kernel size 3, with Dice unchanged to four decimal places (mean voxel agreement 1.000000, lowest case 0.999999, across 27 validation cases on identical trained weights). The outcome is consistent with materialization volume relative to reduction depth setting the cost, and three manipulations agree with that account: dimensionality, materialization strategy and kernel size. Because the two algorithms scale differently in kernel size, their curves cross: over 60 configurations at seven trials each the windowed method wins 26, none of the 12 at kernel size 3 and 11 of 12 at 240 channels with kernel size 5 or above, saving up to 735 ms on a single convolution. A rule reading only layer shape lands within 2.1 percent of a per-shape oracle on that grid and 4.1 percent on held-out shapes, against 13.2 and 16.0 percent (12.6 and 9.8 held out) for the two pure strategies. On whole networks it falls short of cuDNN at kernel size 5 (0.994x) and regresses at kernel size 9 (0.819x), and on a second GPU its advantage over always calling cuDNN on held-out shapes shrinks to 0.01 percent; both are reported rather than tuned away.

Part II takes the cost as close to fixed and asks where to spend it, in MedNeXt with standard operations only. Placing large kernels in the deep stages (3-5-5-7) obtains more receptive field than enlarging every stage (5-5-5-5) at 1.32 against 2.00 times the training step cost, so uniform enlargement is dominated without any accuracy measurement. Trained on MSD Liver, every UpKern placement gains about 0.001 liver Dice over kernel 3, but kernel 3 retrained by the same protocol gains 0.0008, and against that control no placement's difference exceeds the seed spread: on this task the larger kernels add no accuracy the design can resolve.

## Repository Contents

- `proposal.md`, `literature-survey.md`, `novelty-feasibility-audit.md`: the
  project proposal with its survey of related work and its novelty and
  feasibility review; this README and
  `results/` describe what was built and measured.
- `src/`: shape-generic windowed convolution for 2D and 3D, FFT convolution, the
  drop-in `nn.Conv3d` replacement, the shape-aware dispatcher, and the per-stage
  kernel placement utilities. See `src/README.md`.
- `bench/`: correctness verification and the benchmarks behind the reported
  numbers; the Part I measurements made with short inline scripts are recorded
  verbatim in `results/logs/measurements/`.
- `results/`: the measured data those benchmarks produced.
- `requirements.txt`: pinned dependency versions; torch 2.11.0 is the version
  every recorded host ran.

## Headline results

The project has two parts. The first asks whether the convolution itself can be
made cheaper. The second takes the answer as given and asks where the
convolution is worth spending.

**Part I, windowed convolution.** Presented in the literature as a faster
alternative to cuDNN, measured with complete accounting its advantage at
kernel size 3 is small and run-dependent: on the authors' own 3x3 benchmark
layer it falls from 1.56x to 1.13x once the window materialization is timed
(against the fastest of 50 cuDNN trials in a dedicated timing run, 23.8
TFLOPS; across runs cuDNN ranges from 17.6 to 42.4 TFLOPS on that
layer, the top above the kernel's 37.0), and substituted into a whole
3D segmentation network it is 2.34x slower on 3.9x the memory, with Dice
unchanged to four decimal places.

The two algorithms do scale differently in kernel size, and the curves cross.
Across a 60-configuration sweep at seven trials each the windowed method wins in
26: never at kernel size 3 on the median in any of the 12 sweep shapes (one
RTX 3090 trial reached 1.15, and 7 of 8 RTX 3090 held-out kernel size 3 shapes also
lose, the eighth at 1.009), and in 11 of 12 at
240 channels with kernel size 5 or above. Neither pure strategy is close to
optimal, so the useful artifact is a dispatcher rather than a replacement. A
rule reading only layer shape lands within 2.1% of a per-shape oracle on the
grid it was read from, against 13.2% and 16.0% for the pure strategies, and
within 4.1% on held-out shapes, against 12.6% and 9.8%.

That rule then fails in two ways worth reporting. On whole networks it does
not beat cuDNN at kernel size 5 (0.994x, where an autotuner reaches 0.908x) and
regresses at kernel size 9 to 0.819x where an autotuner reaches 1.025x. And on a second
machine, an RTX 4090 host against the RTX 3090 host, 13 of the 54
configurations that two independent sessions on each host agree on change
winner. The clause that looked strongest on the RTX 3090 (240 channels,
kernel size 5 or above, 11 of 12 grid cells) holds in only 8 of 12 grid cells
and 3 of 6 held-out cells on the second, and on held-out shapes the rule's
advantage over simply calling cuDNN falls from 8.25% to 0.01%. A second RTX
4090 host (driver 580.95.05, AMD EPYC 7742, 400 W power limit) changes no
winner against the first in the 57 configurations both hosts settle by the same
two-session standard, while each differs from the RTX 3090 host in 13 or 14.
The change follows the GPU rather than the host, at least on the 4090 side; the
RTX 3090 host was not replicated.

**Part II, kernel placement.** Given that the cost is close to fixed, the
question is where to spend it, and the usual candidate is receptive field.
Cost is very unevenly distributed across stages: enlarging an early-stage kernel
is expensive because it carries the full spatial extent, while a deep stage has
been downsampled repeatedly and is nearly free to enlarge.

Measured on MedNeXt with standard operations only, uniform enlargement is
dominated on cost and receptive field. Placing large kernels deep (3-5-5-7)
obtains more receptive field than enlarging every stage (5-5-5-5), 371 voxels
against 301 in theory and 73 against 67 by the 90 percent mass extent measured
on one random initialization, at lower training step cost, 1.32x against
2.00x. It is better on both axes at once, which needs no accuracy measurement
to establish. FLOPs rank the two the same way but understate the gap, rating
uniform enlargement at 1.21x where the measured training step is 2.00x. Whole
epochs on one RTX 4090 keep the order at a smaller gap: 5-5-5-5 takes 1.64x and
3-5-5-7 (the from-scratch run) 1.11x the epoch time of kernel 3 (the continued
run), from the median epochs of the RTX 4090 rows in the table below, which
also include data loading and validation. Since the trained comparison below
finds no accuracy difference this design can resolve between the placements
and kernel 3 trained for the same total epochs, kernel 3 is the cheapest of
them on FLOPs, latency and step cost, with peak training memory 2.55 GB for
all four.

**Trained comparison: no measurable accuracy gain from the larger kernels.**
Each placement was trained for 150 epochs on fold 0 of MSD Liver, under the
same plans and evaluation, and compared with MedNeXt-B at kernel 3. A
difference counts only if it exceeds the seed spread measured by training one
configuration twice (liver 0.0005, tumour 0.0424, the pair's 0.0407 over the
25 cases both runs score re-expressed over the 24 validation cases that
contain tumour); tumour is scored on those 24 cases. Step cost, FLOPs, inference time and
receptive field come from `results/receptive_field_cost.json` and
`results/effective_receptive_field.json` (RTX 3090; cost at a 64-cubed
patch, batch 2, fp16 autocast; the 90 percent mass extent from the input
gradient of the initialized network at a 96-cubed patch); peak training memory there is 2.55 GB for every placement. Epoch time is
the median over the run's 150 epochs on its own training GPU.

| Configuration | Step cost | FLOPs | Inference | Receptive field (theory / 90% mass) | Liver Dice | vs kernel 3 | Tumour Dice | vs kernel 3 | Epoch time |
|---|---|---|---|---|---|---|---|---|---|
| 3-3-3-3 (kernel 3) | 1.00x | 1.00x | 1.00x | 151 / 59 | 0.9642 | | 0.7099 | | 125.5 s, 3090 |
| 3-3-5-5, UpKern | 1.08x | 1.01x | 1.04x | 271 / 70 | 0.9652 | +0.0010 | 0.7311 | +0.0213 | 107.9 s, 4090 |
| 3-5-5-7, UpKern | 1.32x | 1.06x | 1.22x | 371 / 73 | 0.9653 | +0.0011 | 0.7436 | +0.0338 | 172.9 s, 3090 |
| 5-5-5-5, UpKern | 2.00x | 1.21x | 1.43x | 301 / 67 | 0.9653 | +0.0011 | 0.7564 | +0.0465 | 176.4 s, 4090 |
| 3-5-5-7, from scratch | 1.32x | 1.06x | 1.22x | 371 / 73 | 0.9644 | +0.0002 | 0.7200 | +0.0102 | 119.0 s, 4090 |
| 3-3-3-3, continued by the UpKern protocol | 1.00x | 1.00x | 1.00x | 151 / 59 | 0.9650 | +0.0008 | 0.7508 | +0.0409 | 107.6 s, 4090 |
| nnU-Net | | | | | 0.9546 | -0.0095 | 0.6869 | -0.0230 | 105.8 s, 4090 |

Every UpKern placement gains about 0.001 liver Dice over kernel 3, whatever its
kernels, receptive field or cost, while 3-5-5-7 trained from scratch
gains +0.0002, inside the seed spread. UpKern starts from the trained kernel 3
network, so those runs train for 300 epochs in total against the baseline's
150. MedNeXt's authors ran the matching control on BTCV, kernel 3 retrained
from the trained kernel 3 network, and it gained nothing there (84.00 against
84.01 Dice). On MSD Liver it does not hold: kernel 3 continued the same way,
with nothing enlarged, gains +0.0008, most of the UpKern placements' gain, and
the largest difference between an UpKern placement and it is 0.0003 liver
Dice, inside the seed spread. On tumour, only 5-5-5-5 against kernel 3 exceeds
the seed spread (+0.0465 against 0.0424), and against the continued control it
is +0.0056. On this task, then, a second round of training at kernel 3 matches
the UpKern placements within the seed spread, and the larger kernels add no
accuracy that this design can resolve. The design can resolve a real difference:
MedNeXt-B at kernel 3 beats nnU-Net, trained here for 150 epochs against its
default 1000, by 0.0095 liver Dice, nineteen times the seed spread (paired
Wilcoxon p = 7.4e-05).

That seed spread carries its own caution. The two identically configured runs
that define it differ by 0.0407 tumour Dice, averaged over the 25 validation
cases both runs score, and in this one comparison a paired signed-rank test
across validation cases (two-sided, normal approximation) calls them significantly different at p = 0.049
despite there being no architectural difference to detect; a test across cases
does not see the variation between training runs. A third run of the same
configuration, the kernel 3 baseline above, falls between the pair on both
classes (liver 0.9642 against 0.9639 and 0.9644 over all 27 cases, tumour 0.6815 against 0.6628
and 0.7034, each over 25 scored cases). Single-run ablations on a single fold can manufacture significance
from seed noise, and a replicate is what exposes it here.

## Reproducing

```bash
git clone https://github.com/uday1o1/tensorcore-3d-convolution.git
cd tensorcore-3d-convolution
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python bench/verify_reported_numbers.py   # no GPU: recomputes the derived numbers in its CLAIMED table
python bench/verify_correctness.py     # on a GPU, run this first
python bench/bench_crossover.py 7      # the central result, 7 trials
```

Scripts resolve their own paths, so they run from any working directory.

**What each result needs.** Most of the work reproduces from the clone plus a
CUDA GPU. Only the accuracy arm needs data that is not in the repository: the Medical
Segmentation Decathlon Liver task (CC-BY-SA 4.0), downloaded separately, and a
trained checkpoint.

| Script | Produces | Needs |
|---|---|---|
| `bench/verify_correctness.py` | correctness of every implementation | GPU only |
| `bench/verify_depthwise_indexing.py` | depthwise index validation in float64 against a naive reference | numpy only, no GPU or torch |
| `bench/verify_reported_numbers.py` | checks the derived numbers listed in its `CLAIMED` table against `results/` | nothing, no GPU or torch |
| `bench/bench_crossover.py` | `results/crossover_map_<gpu>.json`, the crossover grid | GPU only |
| `bench/bench_dispatch.py` | `results/dispatch_heldout_<gpu>.json`, held-out rule test | GPU only |
| `bench/bench_training_crossover.py` | `results/training_crossover_<gpu>.json`, forward vs backward vs full step | GPU only |
| `bench/compare_devices.py` | cross-device comparison and prediction audit | nothing, reads `results/` |
| `bench/bench_receptive_field.py` | `results/receptive_field_cost.json`, Part II cost axis | GPU + MedNeXt fork |
| `bench/bench_effective_receptive_field.py` | `results/effective_receptive_field.json`, measured reach | GPU + MedNeXt fork |
| `bench/analyze_variance_pilot.py` | `results/variance_pilot.json`, the seed spread | the validation summaries of two or more seed runs, passed as paths or found under `$RESULTS_FOLDER`; the two pilot summaries behind the committed file are not in the repository, their training logs are (`results/logs/pilot_s1.log`, `pilot_s2.log`) |
| `bench/analyze_training_study.py` | `results/training_study.json`, placement and UpKern comparisons with cost and Dice | nothing, reads `results/validation/` and the committed cost, receptive field and pilot files |
| `bench/bench_reproduce_im2win.py` | the 1.56x and 1.13x reproduction figures | GPU only; the kernel-only figure is a constant measured with `bench/sweep_convs.cpp`, built via `bench/CMakeLists.txt` inside an Im2win clone |
| `bench/bench_end_to_end.py` | whole network substitution cost | GPU + MedNeXt fork |
| `bench/bench_dispatch_network.py` | dispatcher on real networks | GPU + MedNeXt fork |
| `bench/bench_dice_identical_weights.py` | `results/dice_identical_weights.json` | GPU + MedNeXt fork + preprocessed MSD Liver + a trained checkpoint |

The last one takes `--preprocessed` and `--checkpoint`, or reads nnU-Net's own
`$nnUNet_preprocessed` and `$RESULTS_FOLDER`. It reports exactly what is
missing and exits rather than failing obscurely. The dataset is Medical
Segmentation Decathlon Task03 Liver, obtained from its own distributors, and
the checkpoint is produced by training `src/trainers/nnUNetTrainerV2_150ep.py`.
The MedNeXt fork installs from source at the commit `requirements.txt` names.

**Training study.** `scripts/setup_vast.sh train <Trainer>` trains one
configuration from `src/trainers/` on fold 0 and copies its per-case
validation summary to `$BENCH_RESULTS_DIR/validation/`. Train
`nnUNetTrainerV2_MedNeXt_B_k3_150ep_s1` first: the UpKern placements and the
continued control start from its final checkpoint. Copy the summaries into
`results/validation/`, and any regenerated `variance_pilot.json`,
`receptive_field_cost.json` and `effective_receptive_field.json` from
`$BENCH_RESULTS_DIR` into `results/`, before running
`bench/analyze_training_study.py`; it, `compare_devices.py` and
`verify_reported_numbers.py` read only `results/` in the checkout.
The seed spread comes from a separate pair of runs,
`nnUNetTrainerV2_MedNeXt_B_k3_150ep_s1` and `_s2`, trained in their own `WORK`
(run `WORK=<dir> scripts/setup_vast.sh launch setup` there first) so they do
not collide with the baseline, which is a third run of `_s1`: the training
script skips a run whose summary already exists in its results folder. Pass
the pair's two summaries to `bench/analyze_variance_pilot.py`. To train the
UpKern placements on another machine, copy the kernel 3 run's
`model_final_checkpoint.model` there and point `MEDNEXT_UPKERN_SOURCE` at it.

**Environment.** PyTorch 2.11.0+cu128 (CUDA 12.8) with cuDNN 9.19.0 on every
recorded host, with torch pinned in `scripts/setup_vast.sh`; the RTX 3090 measurement
host and the first RTX 4090 benchmark host ran Python 3.12 (paths in
`results/logs/measurements/layout_conversion_shares.log` and
`results/logs/cross4090.log`). Results tagged `rtx3090` were measured on an RTX 3090
and carry no machine fields; the logs in `results/logs/measurements/` come
from one RTX 3090 host (24 GB, driver 595.71.05, recorded in
`results/logs/measurements/host_rtx3090_main_versions.log`); results tagged
`rtx4090` were measured on an RTX 4090 and record its library versions in each
row. The GPU class matters: the RTX 3090 is the one used
in Im2win's own published evaluation, which is what makes the reproduction
comparable. `results/host2/` holds two crossover sessions on a second RTX 4090
host (driver 580.95.05, AMD EPYC 7742, 400 W power limit), which separates the
GPU from the host on the 4090 side. The placement study trained on the hosts
recorded in `results/validation/training_host_rtx3090.txt` (driver 580.126.09,
AMD Ryzen 9 5950X, 320 W limit) and `training_host_rtx4090.txt` (the same recorded
driver, processor and power limit as the second RTX 4090 host); `results/logs/README.md` lists which run used which.

Two sessions of the same convolutions on the same GPU model can disagree. On
the RTX 3090, 4 of 60 cells change winner between sessions, so single cells near
parity should not be read as settled; `bench/compare_devices.py` reports which
cells two sessions agree on. The crossover, held-out and training sweep
scripts alternate which algorithm is timed first, discard warm-up calls, and
record driver, power limit, clocks and host processor with every row; the
files in `results/host2/` carry these fields and the per-order ratios. The `rtx4090` files record the GPU and library versions only, the
`rtx3090` files no machine fields, so that they and the Part I logs came from
one machine is not recorded; "the RTX 3090 host" names the machine or
machines behind those files. Neither set
was measured with alternating order. Across the two `results/host2/` sessions,
which alternate, the order effect on dense cells has a median of 0.999 and
ranges from 0.98 to 1.03.

**Running on a rented instance.** `scripts/setup_vast.sh launch setup` installs
the pinned PyTorch build and the MedNeXt fork, downloads and preprocesses MSD
Task03 Liver, and runs a smoke test, detached and resumable stage by stage. It
is the only file in the repository that knows anything about the instance
layout; it writes `env.sh`, which sets `BENCH_RESULTS_DIR` outside the checkout
so fresh results never block `git pull`. Without that variable, output goes to
`results/`. `scripts/setup_vast.sh launch train <Trainer> ...` trains
configurations in sequence and copies each per-case validation summary out of
the run directory as soon as it finishes.

cuDNN shows real run to run variance on this hardware, so the convolution
sweeps report medians of repeated trials, and whole-network and placement
timings average a timed loop after warm-up. The Im2win reproduction compares
against the fastest of cuDNN's dedicated timing runs (23.8 TFLOPS); a back to
back run of the same shape alongside another measurement reached 42.4. Absolute
timings will differ on other hardware, and so can the crossover: 13 of the 54
settled configurations change winner between the RTX 3090 and RTX 4090 hosts.

## Foundation

This project reimplements the windowed method of `TensorConv/im2win`, the open-source release accompanying its most recent paper, and evaluates it inside `MIC-DKFZ/MedNeXt` and `MIC-DKFZ/nnUNet`, both open-source segmentation frameworks; nnU-Net is under active development, while MedNeXt's repository is stable but has not been updated since late 2024. Full citations and links are in `literature-survey.md`.
