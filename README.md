# Tensor Core Accelerated 3D Convolution for Medical Image Segmentation

**Course:** CMPE 258, Deep Learning, Fall 2026, Prof. Kaikai Liu
**Team:** Uday Arora
**Track:** CMPE 258 has no track menu. This project targets Option 1, Modern Deep Learning Pipeline (Training and Deployment).

## Abstract

Three-dimensional convolutional neural networks are the standard approach for volumetric medical image segmentation, and modern architectures such as MedNeXt rely on large-kernel 3D convolutions whose computational cost is the primary bottleneck to training and inference speed. Im2win, a memory-efficient convolution technique with dedicated Tensor Core support, has been developed and refined across four papers from 2023 to 2026, reporting up to 2.8 times higher throughput than a standard CUDA implementation and 1.4 times higher than cuDNN at reduced memory. Every version of this technique addresses only standard two-dimensional convolution; no published work extends it to three dimensions. This project extends the Im2win windowing technique to 3D convolution, implements it as a PyTorch CUDA extension, and substitutes it for the standard convolution layers inside MedNeXt. The resulting model is trained on a task from the Medical Segmentation Decathlon and compared against nnU-Net and unmodified MedNeXt. Success is defined as a statistically significant reduction in training and inference time and peak memory usage at non-inferior segmentation accuracy, measured by ablation with the convolution implementation as the only varying factor.

## Repository Contents

- `literature-survey.md`: SOTA survey (Deliverable B)
- `proposal.md`: problem formulation, technical approach, device and maintainer (Deliverable C)
- `novelty-feasibility-audit.md`: AI novelty and feasibility audit (Deliverable D)

## Foundation

This project extends `TensorConv/im2win`, the open-source release accompanying its most recent paper, and evaluates it inside `MIC-DKFZ/MedNeXt` and `MIC-DKFZ/nnUNet`, both real, open-source, actively maintained segmentation frameworks. Full citations and links are in `literature-survey.md`.
