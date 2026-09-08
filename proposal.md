# Project Proposal, CMPE 258, Fall 2026

**Title:** Tensor Core Accelerated 3D Convolution for Medical Image Segmentation
**Team:** Uday Arora
**Option:** Option 1, Modern Deep Learning Pipeline (Training and Deployment)

## 1. Problem Formulation

**Domain problem.** Three-dimensional convolutional neural networks are the standard approach for volumetric medical image segmentation. MedNeXt, a current large-kernel 3D ConvNeXt-style architecture, states directly that convolution computation, not attention, is the practical bottleneck for scaling these models. Separately, Im2win, a memory-efficient convolution technique with dedicated Tensor Core support, has been developed across four papers from 2023 to 2026, reporting up to 2.8 times higher throughput than a standard CUDA implementation and 1.4 times higher than cuDNN at 35 to 53 percent of the memory. Every version of Im2win addresses only two-dimensional convolution. A related variant, depthwise convolution, already has dedicated recent CUDA optimization work as of April 2026, but no published work extends the Im2win technique to three dimensions. Three other recent efficiency approaches for 3D medical segmentation were reviewed (see `literature-survey.md`, sources 4 through 6); each addresses efficiency through architecture or representation changes, not through the convolution kernel itself.

**Input and output.** Input: a 3D medical imaging volume from a Medical Segmentation Decathlon task, for example a CT or MRI volume. Output: a voxel-wise segmentation mask over the same volume.

**Target metrics and success criteria, fixed in advance.**

Primary metrics: training throughput and peak GPU memory usage, and inference latency and peak GPU memory usage, for MedNeXt using standard cuDNN 3D convolution versus MedNeXt with the Im2win-based 3D convolution substituted in, all other factors held fixed.

Accuracy constraint: segmentation accuracy, measured by Dice score, must be non-inferior between the two configurations. This project's contribution is efficiency, not accuracy, so no accuracy improvement is claimed or required, only that accuracy is not meaningfully degraded.

Statistical test: a paired Wilcoxon signed-rank test across repeated training runs with different random seeds, with effect size reported as matched-pairs rank-biserial correlation, at least six repetitions per configuration, alpha of 0.05.

Power caveat, stated in advance: a single architecture and dataset and a small number of seeds mean only large effects are reliably detectable; this is disclosed before data collection.

Null-result framing, fixed in advance: if the Im2win-based configuration shows no statistically detected efficiency improvement, or shows a real accuracy degradation, this is reported as exactly that rather than reframed after the fact.

## 2. Proposed Technical Approach

**Data source.** One task from the Medical Segmentation Decathlon, chosen for a bounded and tractable volume count, for example the liver or hippocampus task. The dataset is fully public, with no registration or approval process, available through AWS Open Data, HuggingFace, or native download through the MONAI library.

**Baselines, required by Option 1-C.**

1. nnU-Net, the field's standard self-configuring 3D segmentation framework, run on the chosen task using its own standard configuration.
2. MedNeXt, a current large-kernel 3D ConvNeXt-style architecture, run on the same task using standard cuDNN 3D convolution.

**Technical contribution.** The Im2win windowed convolution technique is extended from two dimensions to three, implemented as a PyTorch CUDA extension following PyTorch's standard custom CUDA extension pattern, and substituted for MedNeXt's standard 3D convolution layers. This is evaluated as an ablation: identical architecture, data, and training procedure, with only the convolution kernel implementation changed, satisfying Option 1-D's efficiency improvement category.

**Implementation plan.**

1. Verify Im2win's windowing scheme against MedNeXt's actual kernel sizes early, since MedNeXt uses larger kernels than Im2win's original two-dimensional evaluation, before committing to full training runs.
2. Extend the windowing computation from two spatial dimensions to three and implement the resulting kernel as a CUDA extension.
3. Validate the extension in isolation, on synthetic 3D convolution shapes, against standard cuDNN 3D convolution for correctness and speed, before integrating it into a full model.
4. Train nnU-Net and standard MedNeXt as baselines on the chosen task.
5. Substitute the extended kernel into MedNeXt and retrain under the identical procedure.
6. Compare all configurations on the metrics defined in Section 1.

## 3. Device Available and Maintainer

**Device available.** A rented A100-class GPU, at a market rate of approximately $1.00 to $1.50 per hour. Im2win's original evaluation used an Ampere-generation GPU, the same architecture family as A100, so no compatibility risk is expected when extending it. Estimated total cost for the training and benchmarking sweep described above: $60 to $200.

**Maintainer.** Uday Arora. Claude Code is used as an AI coding assistant for implementation and experiment scripting, consistent with the course's AI assistance policy.
