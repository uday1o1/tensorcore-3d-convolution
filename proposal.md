# Project Proposal, CMPE 258, Fall 2026

**Title:** Tensor Core Accelerated 3D Convolution for Medical Image Segmentation
**Team:** Uday Arora
**Option:** Option 1, Modern Deep Learning Pipeline (Training and Deployment)

## 1. Problem Formulation

**Why three-dimensional convolution, not two-dimensional.** A CT or MRI scan is a stack of slices, but the structures inside it, a tumor's shape, a vessel's path, do not stop at a slice boundary; they have continuity across the depth axis. A model that processes each slice independently with two-dimensional convolution cannot directly represent that continuity. This is why the field's standard tools are built on true 3D convolution rather than per-slice 2D processing: nnU-Net's own paper states that medical image segmentation is dominated by CNNs built on "2D and 3D vanilla U-Nets," and both baselines used in this project, nnU-Net and MedNeXt, are 3D architectures for exactly this reason.

**The technique this project extends.** Im2win is a memory-efficient convolution method with dedicated Tensor Core support, refined across four papers from 2023 to 2026. Its most recent version reports up to 2.8 times higher throughput than a plain CUDA implementation, 1.4 times higher than cuDNN, and 6.4 times higher than cuBLAS-based convolution, at 35 to 53 percent of the memory, measured on an RTX 3090. All four papers were checked directly, not from memory or abstract, and none mentions three-dimensional or volumetric convolution anywhere. This is the specific gap: a real, working efficiency technique exists, but it has never reached the dimensionality medical segmentation actually requires.

**Why this gap is worth closing, not just unclaimed.** MedNeXt's own text states that "the computational requirements of indefinitely scaling kernel sizes in 3D networks quickly becomes prohibitive," which is why its authors introduce compound scaling instead of simply enlarging kernels. This confirms, in the target architecture's own words, that the operation Im2win optimizes, convolution compute cost, is a real and currently unresolved constraint, not a problem this project is inventing to have something to solve. A related but distinct convolution variant, depthwise convolution, was checked and found to already have dedicated CUDA optimization work as of April 2026, which is why this project targets the three-dimensional gap specifically rather than a variant someone else has already closed. Three further recent papers on efficient 3D medical segmentation were reviewed (`literature-survey.md`, sources 4 through 6); each pursues efficiency through architecture or representation changes, not the convolution kernel, which keeps this project's specific angle open.

**Input and output.** Input: a 3D medical imaging volume from a Medical Segmentation Decathlon task, for example a CT or MRI volume. Output: a voxel-wise segmentation mask over the same volume.

**Target metrics and success criteria, fixed in advance.**

Primary metrics: training throughput and peak GPU memory usage, and inference latency and peak GPU memory usage, for MedNeXt using standard cuDNN 3D convolution versus MedNeXt with the Im2win-based 3D convolution substituted in, all other factors held fixed.

Accuracy constraint: segmentation accuracy, measured by Dice score, must be non-inferior between the two configurations. The contribution claimed here is efficiency, not accuracy, so no accuracy improvement is required, only that accuracy is not meaningfully degraded.

Statistical test: a paired Wilcoxon signed-rank test across repeated training runs with different random seeds, effect size reported as matched-pairs rank-biserial correlation, at least six repetitions per configuration, alpha of 0.05.

Power caveat, stated in advance: one architecture, one dataset, and a small number of seeds mean only large effects are reliably detectable; this is disclosed before data collection, not after a null result.

Null-result framing, fixed in advance: a null or negative result on either efficiency or accuracy is reported as exactly that, not reframed after the fact.

## 2. Proposed Technical Approach

**Data source.** One task from the Medical Segmentation Decathlon, chosen for a bounded and tractable volume count, for example the liver or hippocampus task. Fully public, no registration or approval process, available through AWS Open Data, HuggingFace, or native download through MONAI.

**Baselines, required by Option 1-C.**

1. nnU-Net, run on the chosen task using its own standard, self-configuring setup.
2. MedNeXt, run on the same task using its standard cuDNN-backed 3D convolution.

**How the extension actually works.** Im2win's current implementation processes a 2D input by extracting overlapping rectangular windows so that memory reads stay contiguous, avoiding the memory blow-up of the standard im2col approach. Extending this to 3D means generalizing that windowing from two spatial axes to three: instead of a 2D window sliding over height and width, a 3D window must slide over height, width, and depth simultaneously, which changes both the memory layout the kernel expects and the indexing arithmetic inside it. The resulting kernel is wrapped as a PyTorch CUDA extension, following PyTorch's documented pattern for custom CUDA operators, and substituted for MedNeXt's `nn.Conv3d` layers.

**Evaluation as ablation.** The substitution is evaluated by changing exactly one factor, the convolution kernel implementation, while holding architecture, data, and training procedure fixed. This isolates the effect of the kernel change from every other variable, satisfying Option 1-D's efficiency improvement category.

**Implementation plan.**

1. Check Im2win's windowing scheme against MedNeXt's actual kernel sizes in week one. MedNeXt uses larger kernels than Im2win's original 2D evaluation targeted, and this must be confirmed compatible before full training runs are scheduled.
2. Generalize the windowing computation to three spatial dimensions and implement it as a CUDA extension.
3. Validate the extension in isolation, on synthetic 3D convolution shapes, against standard cuDNN 3D convolution for correctness and speed.
4. Train nnU-Net and standard MedNeXt as baselines on the chosen task.
5. Substitute the extended kernel into MedNeXt and retrain under the identical procedure.
6. Compare all configurations on the metrics defined in Section 1.

## 3. Device Available and Maintainer

Device available: NVIDIA RTX 5050 (Uday Arora). Maintainer: Uday Arora; Claude Code access is requested to support implementation throughout the semester.
