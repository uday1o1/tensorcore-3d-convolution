# AI Novelty and Feasibility Audit: Tensor Core Accelerated 3D Convolution for Medical Image Segmentation

This audit follows two adversarial passes, novelty and feasibility, searching actively for reasons the idea fails or already exists rather than reasons it works.

## AI Critique Summary

Is this genuinely novel? Yes, narrowly. Memory-efficient, Tensor-Core-targeting convolution is not a new idea: it has been developed across four real papers from 2023 to 2026 by the same research group. What has never been done, confirmed by reading all four papers in full text, is extending that specific technique beyond standard two-dimensional convolution. A dedicated search found that the adjacent depthwise-convolution variant already has recent CUDA optimization work, so this project does not target that; it targets three-dimensional convolution, which remains unaddressed. Three other recent efficiency approaches for 3D medical segmentation were found and reviewed, and each pursues efficiency at the architecture or representation level, not the convolution kernel level, which keeps this project's specific angle open. The main risk to disclose honestly: the seed technique's own evaluation is a synthetic microbenchmark with no trained model behind it, so this project's real contribution is closing that gap by integrating the technique into an actual trained, evaluated system, not simply re-running the same benchmark in three dimensions. A second risk: this field moves quickly, a comparable variant was closed within months of its parent paper's publication, so the specific three-dimensional gap should be re-verified immediately before final submission, not assumed stable.

## Novelty Audit: Survives with Named Limitations

**Why it survives.** Four papers spanning 2023 to 2026, all from the same research lineage, were read in full text and confirmed to cover only standard two-dimensional convolution. A dedicated search for the closest adjacent variant, depthwise convolution, found it already addressed by an April 2026 paper, confirming the search methodology can and does find real competing work when it exists. No comparable search result was found for three-dimensional or volumetric extensions of this specific technique.

**Named limitation 1.** This is a narrow, single-lineage extension of one technique to one additional dimensionality, not a general architectural claim, and should be presented that way rather than as a broad "faster 3D deep learning" claim.

**Named limitation 2.** The technique's own published evaluation is a synthetic microbenchmark across twelve layer shapes, with no real dataset or trained model behind it. This project's contribution is specifically closing that gap, and this should be stated directly rather than implied.

**Named limitation 3.** This research area changes on a timescale of weeks rather than months. A final search should be repeated immediately before submission to confirm no closer work has appeared since this audit.

**Update, final independent review.** A fresh, uninvolved reviewer re-checked every claim in this project against primary sources and found the citation work accurate throughout, but identified two real, now-fixed gaps: Option 1-D as originally written claimed only one improvement category (efficiency) against the course's literal "at least two" requirement, and the two original baselines (nnU-Net, MedNeXt) both predate the strict recency window, which the survey itself had already flagged but not resolved. Both are fixed: a second category, an architectural experiment using the efficiency gain's freed compute to test a kernel size MedNeXt's own paper calls prohibitive, is now part of the plan (see proposal.md), and EffiDec3D (CVPR 2025, real code, pushed December 2025) is added as a third, genuinely recent baseline. The reviewer also correctly noted the proposal's stated device (an 8GB desktop GPU) cannot run real training and the plan did not say so plainly; this is now stated directly in proposal.md's Device Available section.

## Feasibility Audit: Survives with Named Limitations

Based only on verified resource facts, not duration estimates.

**Code and technique access.** The Im2win codebase is real, substantive, Apache-2.0 licensed, and actively maintained. Wrapping its CUDA implementation as a PyTorch-compatible extension follows PyTorch's own documented pattern for custom CUDA extensions, a standard and well-established procedure, not a novel engineering risk.

**Baseline access.** Both baselines have real, substantive, permissively licensed code. nnU-Net is actively maintained, last pushed July 2026. MedNeXt's own repository is complete and directly usable but not under active development, last pushed November 2024; its continuation lives in the separately-cited MedNeXt-v2 paper, whose own code release was not verified in this pass. A third candidate baseline considered earlier, a transformer-based architecture named Primus, was checked directly and found to have no released code, and was dropped in favor of MedNeXt.

**Data access.** The Medical Segmentation Decathlon is fully public with no registration, application, or approval process, available through multiple independent channels.

**Compute access.** No GPU architecture compatibility risk is expected, since the technique's original evaluation and this project's target hardware are from the same generation family. Estimated total cost of $60 to $200 for the full training and benchmarking sweep on a rented GPU.

**Named limitation.** Im2win has only ever been tested on 2D kernel shapes (up to size 11 in a single spatial dimension); MedNeXt uses 3D kernels at sizes 3 and 5 per axis. The open question is dimensionality and shape, not size, since the windowing scheme has never been exercised on a genuinely three-dimensional kernel at any size. This should be checked in the first week of implementation, before committing to full-scale training runs.
