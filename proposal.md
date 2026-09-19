# Project Proposal, CMPE 258, Fall 2026

**Title:** Tensor Core Accelerated 3D Convolution for Medical Image Segmentation
**Team:** Uday Arora
**Option:** Option 1, Modern Deep Learning Pipeline (Training and Deployment)

## 1. Problem Formulation

**Why three-dimensional convolution, not two-dimensional.** A CT or MRI scan is a stack of slices, but the structures inside it, a tumor's shape, a vessel's path, do not stop at a slice boundary; they have continuity across the depth axis. A model that processes each slice independently with two-dimensional convolution cannot directly represent that continuity. This is why the field's standard tools are built on true 3D convolution rather than per-slice 2D processing: nnU-Net's own paper states directly that "medical image segmentation is currently dominated by deep convolutional neural networks," and describes nnU-Net itself as "a robust and self-adapting framework on the basis of 2D and 3D vanilla U-Nets." Both baselines used in this project, nnU-Net and MedNeXt, are 3D architectures for exactly this reason.

**The technique this project extends.** Im2win is a memory-efficient convolution method with dedicated Tensor Core support, refined across four papers from 2023 to 2026. Its most recent version reports up to 2.8 times higher throughput than a plain CUDA implementation, 1.4 times higher than cuDNN, and 6.4 times higher than cuBLAS-based convolution, at 35 to 53 percent of the memory, measured on an RTX 3090. All four papers were checked directly, not from memory or abstract, and none mentions three-dimensional or volumetric convolution anywhere. This is the specific gap: a real, working efficiency technique exists, but it has never reached the dimensionality medical segmentation actually requires.

**Why this gap is worth closing, not just unclaimed.** MedNeXt's own text states that "the computational requirements of indefinitely scaling kernel sizes in 3D networks quickly becomes prohibitive," which is why its authors introduce compound scaling instead of simply enlarging kernels. This confirms that convolution compute cost is a real, currently unresolved constraint in 3D medical segmentation generally, not a problem this project is inventing to have something to solve. Three further recent papers on efficient 3D medical segmentation were reviewed (`literature-survey.md`, sources 4 through 6); each pursues efficiency through architecture or representation changes, not the convolution kernel, which keeps this project's specific angle open.

**Which architecture actually has compute to accelerate, checked directly rather than assumed.** Im2win's published kernels assume standard dense convolution, a single reduction over `filter_channels x filter_height x filter_width`. MedNeXt's primary spatial convolution, `conv1` in every block, is depthwise (`groups = in_channels`), and layer-wise FLOP analysis (`FlopCountAnalysis`, MedNeXt-Base, 128 cubed input) confirms it accounts for only 6.2% of the model's total FLOPs: depthwise convolution is cheap by construction, so even a hypothetical zero-cost replacement for `conv1` would cap total-model improvement at roughly that figure, too small a ceiling to justify the kernel engineering involved. The same analysis run on nnU-Net's own plain 3D U-Net, which uses standard dense `Conv3d` throughout with no depthwise layer anywhere, shows 99.6% of its FLOPs are in exactly the dense convolution operations Im2win targets. This project therefore targets nnU-Net's dense 3D convolutions rather than MedNeXt's `conv1`; MedNeXt remains in the project as an unmodified reference baseline (Section 2).

**Input and output.** Input: a 3D medical imaging volume from a Medical Segmentation Decathlon task, for example a CT or MRI volume. Output: a voxel-wise segmentation mask over the same volume.

**Target metrics and success criteria, fixed in advance.**

Primary metrics: training throughput and peak GPU memory usage, and inference latency and peak GPU memory usage, for nnU-Net's 3D U-Net using standard cuDNN 3D convolution versus the same architecture with the Im2win-based 3D convolution substituted into its dense convolution layers, all other factors held fixed.

Accuracy constraint: segmentation accuracy, measured by Dice score, must be non-inferior between the two configurations. The contribution claimed here is efficiency, not accuracy, so no accuracy improvement is required, only that accuracy is not meaningfully degraded.

Statistical test: a paired Wilcoxon signed-rank test across repeated training runs with different random seeds, effect size reported as matched-pairs rank-biserial correlation, at least six repetitions per configuration, alpha of 0.05.

Power caveat, stated in advance: one architecture, one dataset, and a small number of seeds mean only large effects are reliably detectable; this is disclosed before data collection, not after a null result.

Null-result framing, fixed in advance: a null or negative result on either efficiency or accuracy is reported as exactly that, not reframed after the fact.

## 2. Proposed Technical Approach

**Data source and processing.** The Liver task (Task03) from the Medical Segmentation Decathlon: 131 labeled training volumes, plus 70 additional volumes with no public labels (reserved for the Decathlon's own held-out leaderboard, not usable for this project's own accuracy evaluation). Fully public, no registration or approval process, available through AWS Open Data. Preprocessing, resampling, normalization, and patch-size and spacing selection use MedNeXt's own custom nnU-Net-based planning pipeline, applied identically across every configuration below so that the only difference between runs is the factor under test, not a preprocessing inconsistency. Evaluation uses nnU-Net's standard 5-fold cross-validation over the 131 labeled volumes; no custom split is introduced.

**Baselines, required by Option 1-C.**

1. nnU-Net, the field's standard self-configuring 3D segmentation framework, run on the chosen task using its own standard setup with standard cuDNN-backed 3D convolution. This is also the architecture this project modifies (Section 1); the unmodified run is both the required baseline and the direct comparison point for the modified version.
2. MedNeXt, run on the same task using its own standard cuDNN-backed 3D convolution, unmodified. Included as a reference baseline; not the modification target, since its primary spatial convolution was found to carry too small a FLOP share to justify targeting it (Section 1).
3. EffiDec3D (CVPR 2025), a recent optimized decoder for 3D medical segmentation, run using its own released, official code, pushed December 2025. Its repository license is marked "Other" rather than a named permissive license; exact terms are confirmed before its code is used.

**How the extension actually works.** Im2win's current implementation processes a 2D input by extracting overlapping rectangular windows so that memory reads stay contiguous, avoiding the memory blow-up of the standard im2col approach. Extending this to 3D means generalizing that windowing from two spatial axes to three: instead of a 2D window sliding over height and width, a 3D window must slide over height, width, and depth simultaneously, which changes both the memory layout the kernel expects and the indexing arithmetic inside it. Windowed convolution's memory savings come from bounded overlap redundancy, and that redundancy compounds multiplicatively per added spatial axis: a 2D kernel of size 5 rereads each element up to 25 times, while the 3D equivalent rereads it up to 125 times. At nnU-Net's kernel size of 3, this redundancy (27x) is expected to stay within the technique's viable range; at larger kernel sizes, the same analysis puts redundancy at a level where the efficiency advantage over cuDNN is not guaranteed, so any larger kernel size falls back to standard cuDNN 3D convolution if the 3D windowed kernel shows no measured speedup at that size.

Im2win's public repository contains only forward-pass CUDA kernels; no backward pass exists to compute gradients with respect to input or weights. Since nnU-Net is trained end to end, this project implements the corresponding backward kernels and wraps both directions in a custom `torch.autograd.Function`, following PyTorch's documented pattern for custom CUDA operators with manual gradients. This is additional, unavoidable implementation work beyond porting the forward kernel to 3D, and is budgeted into the implementation plan below.

Every convolution in nnU-Net's dense 3D U-Net is already standard (non-depthwise) `Conv3d` at kernel size 3, matching the reduction shape Im2win's kernels assume directly; no architecture change is needed before substitution, unlike an architecture built on depthwise convolution. The resulting kernel is wrapped as a PyTorch CUDA extension and substituted for nnU-Net's dense `Conv3d` layers.

**Two improvement categories, required by Option 1-D.**

1. Efficiency improvement: the kernel substitution itself, evaluated as an ablation changing exactly one factor, the convolution kernel implementation, while holding architecture, data, and training procedure fixed.
2. Architectural change: with the faster kernel's freed compute and memory budget, this project also trains an nnU-Net variant with a larger kernel size than the standard 3, previously more expensive at standard cost, and reports whether that change improves Dice. This turns the efficiency gain into an enabling condition for an architectural experiment, not just a speed measurement on its own.

**Implementation plan.**

1. Determine which candidate architecture's convolutions are actually compute-dominant via layer-wise FLOP analysis before committing kernel-engineering effort to any of them. Verified: MedNeXt's `conv1` is only 6.2% of total FLOPs (depthwise, cheap by construction); nnU-Net's plain 3D U-Net is 99.6% dense `Conv3d` operations. nnU-Net is the target as a result.
2. Generalize Im2win's windowing computation from two spatial axes to three and implement it as a forward CUDA kernel.
3. Implement the corresponding backward kernels and wrap forward and backward in a custom PyTorch autograd function, since Im2win ships no backward pass to build on.
4. Validate the extension in isolation, on synthetic 3D convolution shapes matching nnU-Net's actual per-stage channel widths, against standard cuDNN 3D convolution for correctness, speed, and memory.
5. Train nnU-Net (standard cuDNN 3D convolution) and MedNeXt (unmodified, reference baseline) on the Liver task.
6. Substitute the 3D Im2win kernel into nnU-Net's dense `Conv3d` layers and retrain under the identical procedure.
7. Compare all configurations on the metrics defined in Section 1.

## 3. Device Available and Maintainer

Device available: NVIDIA RTX 5050 (desktop) for code development; full-resolution 3D training at the patch sizes nnU-Net and MedNeXt use exceeds its 8GB memory, so all real training and benchmarking runs on a rented A100-class GPU (approximately $1.00 to $1.50 per hour). Maintainer: Uday Arora; Claude Code access is requested to support implementation throughout the semester.
