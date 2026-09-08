# Tensor Core Accelerated 3D Convolution for Medical Image Segmentation

**Course:** CMPE 258, Deep Learning, Fall 2026, Prof. Kaikai Liu
**Team:** Uday Arora
**Track:** CMPE 258 has no track menu. This project targets Option 1, Modern Deep Learning Pipeline (Training and Deployment).

## Abstract

Medical imaging volumes such as CT and MRI scans have real structure across all three spatial axes, which is why medical image segmentation is dominated by convolutional networks built on true three-dimensional convolutions rather than slice-by-slice two-dimensional processing. Im2win, a memory-efficient convolution technique with dedicated Tensor Core support, has been developed and refined across four papers from 2023 to 2026, reporting up to 2.8 times higher throughput than a standard CUDA implementation and 1.4 times higher than cuDNN at reduced memory, yet every version addresses only two-dimensional convolution, confirmed by a direct full-text search of all four papers. This leaves a real mismatch: the technique that reduces convolution's cost has not reached the dimensionality medical segmentation actually needs. MedNeXt, a current large-kernel 3D architecture, states directly that scaling kernel size in 3D networks quickly becomes computationally prohibitive, confirming that this specific cost is a live constraint in a current model, not a hypothetical one. This project extends Im2win to three dimensions, implements it as a PyTorch CUDA extension, and substitutes it for MedNeXt's standard 3D convolution layers, training on a Medical Segmentation Decathlon task and comparing against nnU-Net and unmodified MedNeXt. Success is defined as a statistically significant reduction in training and inference time and peak memory usage at non-inferior segmentation accuracy.

## Repository Contents

- `literature-survey.md`: SOTA survey (Deliverable B)
- `proposal.md`: problem formulation, technical approach, device and maintainer (Deliverable C)
- `novelty-feasibility-audit.md`: AI novelty and feasibility audit (Deliverable D)

## Foundation

This project extends `TensorConv/im2win`, the open-source release accompanying its most recent paper, and evaluates it inside `MIC-DKFZ/MedNeXt` and `MIC-DKFZ/nnUNet`, both real, open-source segmentation frameworks with complete, usable code; nnU-Net is under active development, while MedNeXt's repository is stable but has not been updated since late 2024. Full citations and links are in `literature-survey.md`.
