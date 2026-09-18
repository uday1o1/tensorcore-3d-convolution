# AI Novelty and Feasibility Audit: Tensor Core Accelerated 3D Convolution for Medical Image Segmentation

This audit follows two adversarial passes, novelty and feasibility, searching actively for reasons the idea fails or already exists rather than reasons it works.

## AI Critique Summary

Is this genuinely novel? Yes, narrowly. Memory-efficient, Tensor-Core-targeting convolution is not a new idea: it has been developed across four real papers from 2023 to 2026 by the same research group. What has never been done, confirmed by reading all four papers in full text, is extending that specific technique beyond standard two-dimensional convolution. Three other recent efficiency approaches for 3D medical segmentation were found and reviewed, and each pursues efficiency at the architecture or representation level, not the convolution kernel level, which keeps this project's specific angle open. Direct inspection of both codebases surfaced a second, independent gap on top of the 2D-to-3D one: MedNeXt's target layer is depthwise, while Im2win's kernels assume dense convolution, so this project converts that layer to dense as part of its method rather than as an incidental detail. The main risk to disclose honestly: the seed technique's own evaluation is a synthetic microbenchmark with no trained model behind it, so this project's real contribution is closing that gap by integrating the technique into an actual trained, evaluated system, not simply re-running the same benchmark in three dimensions. A second risk: this field moves quickly, a comparable variant was closed within months of its parent paper's publication, so the specific three-dimensional gap should be re-verified immediately before final submission, not assumed stable.

## Novelty Audit: Survives with Named Limitations

**Why it survives.** Four papers spanning 2023 to 2026, all from the same research lineage, were read in full text and confirmed to cover only standard two-dimensional convolution. A dedicated search for the closest adjacent variant, depthwise convolution, found it already addressed by an April 2026 paper, confirming the search methodology can and does find real competing work when it exists; that paper optimizes depthwise convolution directly, a different problem from converting a depthwise layer to dense and then applying a dense-convolution technique, which is this project's actual method. No comparable search result was found for three-dimensional or volumetric extensions of Im2win specifically.

**Named limitation 1.** This is a narrow, single-lineage extension of one technique to one additional dimensionality, not a general architectural claim, and should be presented that way rather than as a broad "faster 3D deep learning" claim.

**Named limitation 2.** The technique's own published evaluation is a synthetic microbenchmark across twelve layer shapes, with no real dataset or trained model behind it. This project's contribution is specifically closing that gap, and this should be stated directly rather than implied.

**Named limitation 3.** This research area changes on a timescale of weeks rather than months. A search for closer work is repeated immediately before submission.

**Named limitation 4.** Efficiency alone is one improvement category; the project also includes an architectural experiment using the efficiency gain's freed compute to test a kernel size MedNeXt's own paper calls prohibitive, satisfying the two-category requirement with a change genuinely enabled by the first.

## Feasibility Audit: Survives with Named Limitations

Based only on verified resource facts, not duration estimates.

**Code and technique access.** The Im2win codebase is real, substantive, Apache-2.0 licensed, and actively maintained. Its repository contains only forward-pass CUDA kernels; no backward pass exists for training. Wrapping a working forward kernel as a PyTorch extension is a standard, well-established procedure, but this project also has to design and implement the backward kernels itself and wrap both directions in a custom autograd function, since MedNeXt is trained end to end and no reference backward implementation exists to build from. This is real engineering risk beyond the 2D-to-3D port, and is treated as such rather than folded into the "standard wrapping" claim.

**Baseline access.** All three baselines have real, substantive code. nnU-Net is actively maintained, last pushed July 2026. MedNeXt's repository is complete and directly usable but not under active development, last pushed November 2024. EffiDec3D's repository is official and was pushed as recently as December 2025, satisfying the course's recency requirement; its license, marked "Other," is confirmed before use.

**Data access.** The Medical Segmentation Decathlon is fully public with no registration, application, or approval process, available through multiple independent channels.

**Compute access.** No GPU architecture compatibility risk is expected, since the technique's original evaluation and this project's target hardware are from the same generation family. Estimated total cost of $60 to $200 for the full training and benchmarking sweep on a rented GPU.

**Named limitation 1.** Im2win has only ever been tested on 2D kernel shapes (up to size 11 in a single spatial dimension); MedNeXt uses 3D kernels at sizes 3 and 5 per axis. The open question is dimensionality and shape, not size, since the windowing scheme has never been exercised on a genuinely three-dimensional kernel at any size. This should be checked in the first week of implementation, before committing to full-scale training runs.

**Named limitation 2.** Converting MedNeXt's `conv1` from depthwise to dense convolution changes the layer's parameter count and receptive-field mixing behavior, not just its kernel implementation. The efficiency comparison is therefore run against a dense-MedNeXt baseline (standard cuDNN, dense `conv1`), not the original depthwise MedNeXt, so the ablation isolates the convolution kernel implementation as the only varying factor. Whether dense-MedNeXt's own accuracy differs materially from the original depthwise MedNeXt is a secondary question this project also reports, since it affects how the results should be read against the published MedNeXt baseline.

**Named limitation 3.** The backward-kernel implementation (see Feasibility, Code and technique access) has no existing reference to validate against beyond standard gradient-checking against PyTorch's autograd on small synthetic inputs. This is the single highest-risk implementation step in the project and is scheduled early, before any full training run depends on it.
