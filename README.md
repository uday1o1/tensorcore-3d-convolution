# Tensor Core Accelerated 3D Convolution for Medical Image Segmentation

**Course:** CMPE 258, Deep Learning, Fall 2026, Prof. Kaikai Liu
**Team:** Uday Arora
**Option:** Option 1, Modern Deep Learning Pipeline (Training and Deployment)

## Abstract

Medical image segmentation is dominated by networks built on true three-dimensional convolution, and the cost of that convolution is a live constraint: MedNeXt, a current large-kernel 3D architecture, states directly that scaling kernel size in 3D quickly becomes computationally prohibitive. Im2win, a memory-efficient windowed convolution with Tensor Core support refined across four papers from 2023 to 2026, reports up to 1.4 times higher throughput than cuDNN at reduced memory, but every version addresses only two dimensions. This project set out to extend it to three dimensions and substitute it into a volumetric segmentation network, expecting a speedup.

That expectation did not survive contact with measurement, and the investigation redirected accordingly. Two findings forced the change. First, the published comparison times the convolution kernel alone while the window materialization the method requires runs outside the timing loop, whereas cuDNN's implicit GEMM performs the equivalent work inside its timed kernel; implementing that materialization on the GPU and charging for it moves a reproduced 1.56 times advantage to 1.13. Second, the released kernels are specialized to individual benchmark layers through hardcoded launch geometry and return silently zero or partially computed output at any other problem shape, so there was no general implementation to port.

We therefore built a shape-generic windowed convolution for two and three dimensions, verified against cuDNN on shapes the released kernels cannot execute, and used it to characterize when the approach pays off rather than to assert that it does. Substituted into a whole segmentation network at the standard kernel size of 3, it is 2.34 times slower on 3.9 times the memory, at numerically identical accuracy (voxel agreement 1.000000 across 27 validation cases on identical trained weights). Materialization volume relative to reduction depth governs the outcome, confirmed by three independent manipulations: dimensionality, materialization strategy, and kernel size.

The same mechanism locates a regime the literature does not benchmark. Because implicit GEMM scales with the `k^3` growth in work while windowed materialization grows only linearly in `k`, the curves cross. Across a 60-configuration sweep the windowed method wins in 27, never at kernel size 3 and always at 240 channels once the kernel reaches size 5, saving up to 780 ms on a single convolution. The practical conclusion is not that one algorithm should replace the other but that the choice is shape-dependent and predictable, so the useful artifact is a dispatcher. The deliverables on file (`proposal.md`, `literature-survey.md`, `novelty-feasibility-audit.md`) describe the original framing; this README and `results/` describe what was measured.

## Repository Contents

- `literature-survey.md`: SOTA survey (Deliverable B)
- `proposal.md`: problem formulation, technical approach, device and maintainer (Deliverable C)
- `novelty-feasibility-audit.md`: AI novelty and feasibility audit (Deliverable D)
- `src/`: shape-generic windowed convolution for 2D and 3D, FFT convolution, the
  drop-in `nn.Conv3d` replacement, and the shape-aware dispatcher. See `src/README.md`.
- `bench/`: correctness verification and every benchmark behind the reported numbers.
- `results/`: the measured data those benchmarks produced.
- `requirements.txt`: pinned versions the measurements were taken with.

## Headline result

Windowed convolution is usually presented as a faster alternative to cuDNN. Measured
with complete accounting it is not, at the kernel sizes everyone benchmarks. It is
2.34 times slower than cuDNN substituted into a whole 3D segmentation network, on 3.9
times the memory, at numerically identical accuracy.

But the two algorithms scale differently in kernel size. Implicit GEMM scales with the
`k^3` growth in work, while windowed materialization grows only linearly in `k`. The
curves cross. Across a 60-configuration sweep the windowed method wins in 27. Two rules
hold across the whole grid: it never wins at kernel size 3, the size essentially every
published convolution benchmark uses, and it always wins at 240 channels once the kernel
reaches size 5. Speedups reach 2.36x, and on the largest layer measured it saves 780 ms
on a single convolution.

The regime in which this method is evaluated is the regime in which it is worst.

## Reproducing

```bash
git clone https://github.com/uday1o1/tensorcore-3d-convolution.git
cd tensorcore-3d-convolution
pip install -r requirements.txt
python bench/verify_correctness.py     # run this first
python bench/bench_crossover.py        # the central result
```

Scripts resolve their own paths, so they run from any working directory.

**What each result needs.** Most of the work reproduces from the clone plus a
CUDA GPU. Only the accuracy arm needs data we cannot redistribute.

| Script | Produces | Needs |
|---|---|---|
| `bench/verify_correctness.py` | correctness of every implementation | GPU only |
| `bench/verify_depthwise_indexing.py` | depthwise index validation in exact arithmetic | nothing, no GPU or torch |
| `bench/bench_crossover.py` | `results/crossover_map.json`, the crossover grid | GPU only |
| `bench/bench_dispatch.py` | `results/dispatch_heldout.json`, held-out rule test | GPU only |
| `bench/bench_reproduce_im2win.py` | the 1.56x and 1.13x reproduction figures | GPU; the kernel-only number additionally needs Im2win built via `bench/CMakeLists.txt` |
| `bench/bench_end_to_end.py` | whole network substitution cost | GPU + MedNeXt fork |
| `bench/bench_dispatch_network.py` | dispatcher on real networks | GPU + MedNeXt fork |
| `bench/bench_dice_identical_weights.py` | `results/dice_identical_weights.json` | GPU + MedNeXt fork + preprocessed MSD Liver + a trained checkpoint |

The last one takes `--preprocessed` and `--checkpoint`, or reads nnU-Net's own
`$nnUNet_preprocessed` and `$RESULTS_FOLDER`. It reports exactly what is
missing and exits rather than failing obscurely. The dataset is Medical
Segmentation Decathlon Task03 Liver, obtained from its own distributors, and
the checkpoint is produced by training `src/trainers/nnUNetTrainerV2_150ep.py`.

**Environment.** Measured on NVIDIA RTX 3090 (24GB), sm_86, CUDA 12.8, PyTorch
2.11.0+cu128, Python 3.12.14, cuDNN 9.19.0. The GPU class matters: it is the
same one used in Im2win's own published evaluation, which is what makes the
reproduction comparable. Results in `results/` were produced across two RTX
3090 hosts, on drivers 595.71.05 and 570.86.15.

cuDNN shows real run to run variance on this hardware, so all results are
medians of repeated trials, taken against the cuDNN figure most favorable to
cuDNN. Absolute timings will differ on other hardware; the ratios and the
crossover structure are the claims.

## Foundation

This project extends `TensorConv/im2win`, the open-source release accompanying its most recent paper, and evaluates it inside `MIC-DKFZ/MedNeXt` and `MIC-DKFZ/nnUNet`, both real, open-source segmentation frameworks with complete, usable code; nnU-Net is under active development, while MedNeXt's repository is stable but has not been updated since late 2024. Full citations and links are in `literature-survey.md`.
