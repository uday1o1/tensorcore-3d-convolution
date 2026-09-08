# Literature and SOTA Survey: Tensor Core Accelerated 3D Convolution for Medical Image Segmentation

Sources below were retrieved and read directly. Verification level, full text or abstract only, is stated per source.

### 0. Fu, Ma, Zhang, Zhao, Lu, Liu. "Enabling Memory-efficient Im2win Convolution with Multi-precision Support on GPU CUDA and Tensor Cores." IJCNN 2026. [arXiv:2608.20725](https://arxiv.org/abs/2608.20725). Code: [github.com/TensorConv/im2win](https://github.com/TensorConv/im2win)

Read in full text. Primary technique this project extends. Im2win reduces the memory overhead of standard im2col-based convolution by processing input in overlapping windows with contiguous memory access, and this version adds full-precision CUDA core execution and half-precision Tensor Core execution with a double-buffered prefetch scheme. Reports up to 2.8 times higher throughput than a CUDA core baseline, 1.4 times higher than cuDNN, and 6.4 times higher than cuBLAS-based convolution, at 35 to 53 percent of the memory. Tested only on standard two-dimensional convolution and only on one GPU, an RTX 3090.

### 0b. Lu et al. "Im2win: An Efficient Convolution Paradigm on GPU." [arXiv:2306.14316](https://arxiv.org/abs/2306.14316); "Im2win: Memory Efficient Convolution on SIMD Architectures." [arXiv:2306.14320](https://arxiv.org/abs/2306.14320); "High Performance Im2win and Direct Convolutions Using Three Tensor Layouts on SIMD Architectures." [arXiv:2408.00278](https://arxiv.org/abs/2408.00278)

Read in full text. The full lineage of this technique from 2023 to 2024, prior to the 2026 Tensor Core version above. All three, together with source 0, cover only two-dimensional convolution. No transposed, grouped, depthwise, or three-dimensional convolution appears anywhere across four papers spanning three years, which is the specific gap this project addresses. Cited as necessary background, not as recent SOTA subject to the course's one to two year recency requirement.

### 1. "CUDA Kernel Optimization and Counter-Free Performance Analysis for Depthwise Convolution in Cloud Environments." [arXiv:2604.25422](https://arxiv.org/abs/2604.25422), April 2026

Read at abstract level. Confirms that depthwise convolution, an adjacent variant to the one this project targets, already has dedicated recent CUDA kernel optimization work. This project does not target depthwise convolution; it targets three-dimensional convolution, which this source does not cover.

### 2. Roy et al. "MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation." MICCAI 2023. [arXiv:2303.09975](https://arxiv.org/abs/2303.09975). Code: [github.com/MIC-DKFZ/MedNeXt](https://github.com/MIC-DKFZ/MedNeXt)

Read at abstract level. Baseline model and the primary target for the convolution kernel substitution. MedNeXt is a ConvNeXt-style architecture built on large-kernel three-dimensional convolutions, whose own stated motivation is that convolution cost, not attention, is the practical bottleneck for scaling 3D segmentation networks. This makes it a directly motivated target for a memory-efficient convolution kernel, rather than an arbitrary choice.

### 2b. "MedNeXt-v2." [arXiv:2512.17774](https://arxiv.org/abs/2512.17774), December 2025

Read at abstract level. A recent continuation of the MedNeXt family, confirming the architecture is still actively developed and current within the course's recency window.

### 3. Isensee et al. "nnU-Net: A Self-configuring Method for Deep Learning-based Biomedical Image Segmentation." Nature Methods, 2021. Code: [github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)

Read at abstract level. Second baseline. Still the field's standard reference framework for medical segmentation and under active maintenance, though the original paper predates the course's recency window; it is included as a required, currently-maintained baseline, not as a recent SOTA claim.

### 4. "Johnson-Lindenstrauss Lemma Guided Network for Efficient 3D Medical Segmentation." [arXiv:2509.22307](https://arxiv.org/abs/2509.22307), September 2025

Read at abstract level. Pursues efficiency in 3D segmentation through a dimensionality-reduction-guided architecture rather than a convolution kernel change. Evidence that efficient 3D medical segmentation is an active area, approached here at the architecture level rather than the GPU kernel level.

### 5. "TokenSeg: Efficient 3D Medical Image Segmentation via Hierarchical Visual Token Compression." [arXiv:2601.04519](https://arxiv.org/abs/2601.04519), January 2026

Read at abstract level. Pursues efficiency through sparse token representation rather than a convolution implementation change. Same pattern as source 4: efficiency addressed at a different level of the system than this project's contribution.

### 6. Rahman and Marculescu. "EffiDec3D: An Optimized Decoder for High-Performance and Efficient 3D Medical Image Segmentation." CVPR 2025.

Read at abstract level. Pursues efficiency through decoder redesign. Together with sources 4 and 5, this establishes that efficient 3D medical segmentation is a genuinely active 2025-2026 research area, with no surveyed source pursuing the specific angle of a memory-efficient, Tensor-Core-targeting convolution kernel.

### 7. Antonelli et al. "The Medical Segmentation Decathlon." Nature Communications, 2022.

Dataset paper. The Medical Segmentation Decathlon is the data source for this project, fully public with no registration requirement.

---

**Summary of the gap.** A memory-efficient, Tensor-Core-targeting convolution technique exists and is actively developed, but has never been extended beyond two-dimensional convolution across four papers and three years. Three-dimensional medical segmentation efficiency is an active research area, but every recent approach found addresses it at the architecture or representation level, not at the level of the convolution kernel itself. This project sits at the intersection: extending a real, working kernel-level technique to the dimensionality where it has not yet been applied, inside a real, currently-relevant architecture whose own stated bottleneck is exactly the operation this technique targets.
