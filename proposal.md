# Project Proposal, CMPE 258, Fall 2026

**Title:** Does Windowed Convolution Scale to Three Dimensions? Reproduction, Honest Accounting, and Dimensional Extension of Im2win for Medical Image Segmentation
**Team:** Uday Arora
**Option:** Option 1, Modern Deep Learning Pipeline (Training and Deployment)

## 1. Problem Formulation

**The family of techniques this project examines.** Convolution on GPUs is dominated by two strategies: implicit GEMM, used by cuDNN, which streams data directly into matrix-multiply hardware without ever materializing an intermediate buffer; and the im2col family, which first rearranges the input into a matrix and then calls a dense matrix multiply. Im2win is the current refinement of the second family, replacing im2col's full duplication with a window layout that keeps memory reads contiguous and adds Tensor Core support. It has been developed across four papers from 2023 to 2026 and reports up to 1.4 times higher throughput than cuDNN at 35 to 53 percent of the memory.

**A discrepancy found by direct reproduction.** This project began by reproducing Im2win's published comparison on the same GPU class the original evaluation used, an RTX 3090. The kernel result reproduces and is genuinely strong: on the AlexNet-style layer the authors label conv9, Im2win's Tensor Core kernel sustains 37.1 TFLOPS against cuDNN's 23.8 TFLOPS, a ratio of 1.56, exceeding the published claim. However, inspection of the benchmark harness shows that the reported timing covers the convolution kernel alone. The `image2window` transform that produces the kernel's input is executed on the host, outside the timing loop, and reused across all fifty timed iterations, while allocation and transfer costs are accumulated into a separate variable that is not reported. cuDNN's implicit GEMM performs the equivalent rearrangement inside its kernel and is timed in full. Restoring the transform cost, measured at its bandwidth-bound floor of 0.739 ms (835 GB/s achieved, roughly 89 percent of the device's theoretical peak), changes the end-to-end ratio from 1.56 to **1.04**, which is parity.

**Why this matters, and what question it opens.** The materialization that the transform performs is not an implementation detail that better engineering removes; it is intrinsic to the im2col family, which is precisely what implicit GEMM avoids. This offers a mechanistic explanation for something otherwise puzzling: why windowed convolution approaches continue to report kernel-level wins without displacing implicit GEMM in production libraries. It also raises a question that no published work answers. Window overlap redundancy compounds multiplicatively with each spatial axis, so a third dimension simultaneously makes the kernel more compute-dense (the reduction depth grows from `C x 9` to `C x 27` at kernel size 3) and makes the buffer it depends on more expensive to materialize. Whether that trade nets positive or negative is unknown, and it determines whether the windowed paradigm has a future in volumetric workloads.

**Why three-dimensional convolution is the right setting for that question.** Medical imaging volumes such as CT and MRI scans have structure across all three axes, which is why the field's standard tools are built on true 3D convolution rather than per-slice 2D processing. nnU-Net's own paper states that "medical image segmentation is currently dominated by deep convolutional neural networks," and describes nnU-Net as "a robust and self-adapting framework on the basis of 2D and 3D vanilla U-Nets." MedNeXt's text states that "the computational requirements of indefinitely scaling kernel sizes in 3D networks quickly becomes prohibitive." Volumetric convolution is therefore both a workload the field genuinely runs and one where convolution cost is an acknowledged constraint.

**Input and output.** Input: a 3D medical imaging volume from a Medical Segmentation Decathlon task. Output: a voxel-wise segmentation mask over the same volume.

**Contributions claimed.**

1. A reproduction of Im2win's published 2D comparison with full accounting, establishing that the kernel-level advantage is real (1.56 times) but that the end-to-end advantage under honest accounting is parity (1.04 times) on the authors' own benchmark shapes and GPU class.
2. The first extension of the windowing scheme to three dimensions, used to test a stated prediction: that the advantage degrades further with dimensionality because materialization cost grows faster than the compute-density benefit.
3. An empirical characterization of where windowed convolution does and does not pay off in a real volumetric segmentation workload, including a measured accounting of where GPU time actually goes in the standard stack.

**Target metrics and success criteria, fixed in advance.**

Primary measurement: kernel-level throughput and peak memory for windowed convolution against cuDNN, reported both with and without the materialization cost included, across the convolution shapes nnU-Net actually uses, in both two and three dimensions.

Secondary measurement: end-to-end inference latency and peak memory for nnU-Net using standard cuDNN convolution versus the same network with the windowed kernel substituted into its dense convolution layers.

Accuracy constraint: segmentation accuracy, measured by Dice score, must be non-inferior between kernel configurations. Because the substitution is performed on identical trained weights, any accuracy difference is attributable to the kernel's numerical behavior alone rather than to training variance.

Statistical test: paired one-sided Wilcoxon signed-rank tests over per-volume results, with the validation volumes of each cross-validation fold as the paired units. One-sided tests are appropriate because both claims are directional: non-inferiority for accuracy, and superiority for speed. Effect size reported as matched-pairs rank-biserial correlation, alpha of 0.05.

Success criterion, stated honestly: this project succeeds if it produces a defensible characterization of when windowed convolution is and is not competitive, not if a particular speedup is achieved. Given the reproduction result above, the expected outcome is that the windowed approach does not beat cuDNN end-to-end in three dimensions. That prediction is stated before the experiment rather than after it.

Null-result framing, fixed in advance: a null or negative result on efficiency is the anticipated outcome and is reported as exactly that, with its mechanism, not reframed after the fact.

## 2. Proposed Technical Approach

**Data source and processing.** The Liver task (Task03) from the Medical Segmentation Decathlon: 131 labeled training volumes, plus 70 additional volumes with no public labels, reserved for the Decathlon's own held-out leaderboard and therefore not usable for this project's accuracy evaluation. The task has three classes: background, liver, and cancer. Fully public, no registration or approval process, available through AWS Open Data. Preprocessing, resampling, normalization, and patch size and spacing selection use MedNeXt's custom nnU-Net-based planning pipeline, which fixes 1mm isotropic spacing and a 128 cubed patch size, applied identically across every configuration so that the only difference between runs is the factor under test. Evaluation uses nnU-Net's standard 5-fold cross-validation over the 131 labeled volumes; no custom split is introduced.

**Baselines, required by Option 1-C.**

1. nnU-Net, the field's standard self-configuring 3D segmentation framework, run using its own standard setup with cuDNN convolution. This is also the architecture into which the windowed kernel is substituted; the unmodified run is both the required baseline and the direct comparison point.
2. MedNeXt, run unmodified on the same task as a reference architecture. It is not the substitution target: layer-wise FLOP analysis showed its primary spatial convolution is depthwise and accounts for only 6.2 percent of total FLOPs, too small a share to be informative, whereas nnU-Net's plain 3D U-Net carries 99.6 percent of its FLOPs in the dense convolutions the technique addresses.
3. EffiDec3D (CVPR 2025), a recent optimized decoder for 3D medical segmentation, run using its own released official code as a reference point. Its UT Austin Research License permits academic and research use but prohibits redistribution, so its code is cloned by the user and never vendored into this repository.

**The experimental matrix.** Because Im2win already works in two dimensions, the 2D configuration is not a competing alternative to this work but its control condition. Four cells are measured under identical data, framework, and hardware:

| | cuDNN | Windowed (Im2win) |
|---|---|---|
| 2D (nnU-Net `2d` configuration) | baseline | existing kernel, unmodified |
| 3D (nnU-Net `3d_fullres`) | baseline | this project's extension |

Without the 2D arm, a 3D result has nothing to be interpreted against. With it, the dimensional scaling of the advantage is measured directly rather than asserted, which is the central question stated in Section 1.

**How the extension works.** Im2win processes a 2D input by extracting overlapping windows so that memory reads stay contiguous, avoiding the memory expansion of standard im2col. Extending this to 3D means generalizing the windowing from two spatial axes to three, changing both the memory layout the kernel expects and the indexing arithmetic inside it. Every convolution in nnU-Net's 3D U-Net is already standard dense `Conv3d` at kernel size 3, matching the reduction shape the kernels assume, so no architecture change is required before substitution. Two layer types are excluded from substitution on shape grounds: the first convolution, whose reduction depth of 27 is not a multiple of the Tensor Core tile depth of 16, and the 1x1x1 segmentation heads, which are not spatial convolutions. Together these account for under one percent of convolution FLOPs.

**Scope of the kernel work.** The published repository contains forward-pass CUDA kernels only, with no backward pass and no PyTorch binding of any kind. The primary experiments therefore substitute the kernel at inference time on identically trained weights, which removes the backward pass from the critical path and produces a cleaner comparison, since the two configurations differ in nothing but the convolution implementation. Backward kernels and training-time substitution are a secondary objective, pursued only if the inference results warrant it.

**Two improvement categories, required by Option 1-D.**

1. Efficiency: the kernel substitution itself, evaluated as an ablation changing exactly one factor, the convolution implementation, while holding architecture, weights, data, and precision fixed.
2. Measurement and accounting: a full profile of where GPU time is actually spent in the standard 3D segmentation stack, which established that convolution accounts for 65.5 percent of inference time and that layout conversion between NCHW and NHWC consumes a further 12 percent. The latter is a previously unstated inefficiency and is directly relevant, since the windowed kernels are natively NCHW and would not require that conversion.

**Implementation plan.**

1. Reproduce Im2win's published 2D benchmark with full accounting, separating kernel time from materialization cost. (Completed; see Section 1.)
2. Profile the standard stack to establish where time is actually spent and what fraction any convolution kernel can address. (Completed; convolution is 65.5 percent of inference time, layout conversion a further 12 percent.)
3. Benchmark the existing 2D windowed kernel against cuDNN on nnU-Net's real 2D layer shapes, establishing the control arm.
4. Generalize the windowing computation to three spatial axes and implement it as a forward CUDA kernel.
5. Validate the 3D kernel in isolation against cuDNN for correctness, throughput, and memory, at nnU-Net's actual per-stage channel widths, reporting results both with and without materialization cost.
6. Expose the validated kernel through a PyTorch custom operator and substitute it into nnU-Net's dense convolution layers.
7. Train nnU-Net in both 2D and 3D configurations, plus the reference baselines, and measure inference latency, memory, and Dice for each kernel on identical weights.
8. Report the dimensional scaling result and its mechanism, in whichever direction it falls.

## 3. Device Available and Maintainer

Device available: NVIDIA RTX 5050 (desktop) for code development; full-resolution 3D training at the patch sizes nnU-Net and MedNeXt use exceeds its 8GB memory, so all real training and benchmarking runs on a rented RTX 3090, matching the GPU class used in Im2win's own published evaluation, which makes the reproduction directly comparable. Maintainer: Uday Arora; Claude Code access is requested to support implementation throughout the semester.
