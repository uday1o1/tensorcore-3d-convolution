# AI Novelty and Feasibility Audit: Windowed Convolution in Three Dimensions

This audit follows two adversarial passes, novelty and feasibility, searching actively for reasons the idea fails or already exists rather than reasons it works. It reflects the project as reframed after a direct reproduction of Im2win's published 2D comparison changed what the defensible contribution is.

## AI Critique Summary

Is this genuinely novel? Yes, and more defensibly than the original framing. The original claim was that extending a memory-efficient Tensor Core convolution technique from 2D to 3D would accelerate medical image segmentation. Reproducing the seed technique's own benchmark on its own GPU class first, rather than assuming its headline number, changed that picture: the kernel result reproduces and is strong (37.1 TFLOPS against cuDNN's 23.8, a ratio of 1.56, above the published 1.4), but the published timing covers the convolution kernel alone. The `image2window` transform that produces the kernel's input runs on the host outside the timing loop, while cuDNN's implicit GEMM performs the equivalent work inside its timed kernel. Restoring that cost at its bandwidth-bound floor moves the end-to-end ratio to 1.04, which is parity. The contribution is therefore no longer a speedup claim. It is a reproduction with honest accounting, a mechanistic explanation for why windowed convolution keeps reporting kernel wins without displacing implicit GEMM in production libraries, and a dimensional-scaling test of whether the approach can work in 3D at all. The main risk to disclose: the expected outcome is now a negative result on raw speed, which is stated in advance rather than discovered and reframed later.

## Novelty Audit: Survives with Named Limitations

**Why it survives.** Three separable claims, each checked.

First, the 2D-to-3D extension of this windowing scheme is unclaimed. Four papers spanning 2023 to 2026 were read in full text and cover only two-dimensional convolution; no transposed, grouped, depthwise, or volumetric variant appears anywhere across them.

Second, the accounting correction is unclaimed and verified by direct measurement rather than argument. No published source states that Im2win's advantage is consumed by materialization cost.

Third, a dedicated search for competing work on the new target found none. The nearest result is NVIDIA's own optimized nnU-Net for PyTorch, which uses standard automatic mixed precision and TF32 on ordinary `Conv3d` calls: that is the cuDNN baseline this project measures against, not a competing windowed implementation. A second search for windowed convolution or Im2win in medical image segmentation returned no three-dimensional or segmentation-specific application; the single tangential result is a 2026 ARM CPU port, a different axis of extension. The search methodology's ability to find real competing work when it exists was established earlier, when it surfaced an April 2026 depthwise-convolution CUDA optimization paper.

**Named limitation 1.** This is a narrow, single-lineage examination of one technique family, not a general claim about GPU convolution, and should be presented that way.

**Named limitation 2.** The reproduction result rests on a transform cost measured as a bandwidth-bound floor using a proxy, not on the authors' own transform kernel instrumented in place. The proxy achieved 835 GB/s, roughly 89 percent of the device's theoretical peak, so it is a realistic floor rather than a pessimistic estimate, and materializing the windowed buffer is intrinsic to the design rather than avoidable by better engineering. Even so, the specific figure of 1.04 should be reported as a floor-based estimate with its method stated, not as a measurement of the authors' implementation.

**Named limitation 3.** The expected result on raw speed is negative. A project whose anticipated finding is that a technique does not beat the incumbent must be judged on the quality of its characterization and mechanism, not on a performance number. This is stated before data collection.

**Named limitation 4.** This research area changes on a timescale of weeks. A search for closer work is repeated immediately before submission.

**Named limitation 5.** The choice of nnU-Net as the substitution target follows from a compute-share argument (99.6 percent of its FLOPs are in dense `Conv3d`, verified) rather than from architectural interest. nnU-Net's plain 3D U-Net is an older, less parameter-efficient design than MedNeXt. The contribution is about convolution implementation, not segmentation architecture, and should be presented that way.

**Named limitation 6.** A two-dimensional configuration of the same network is a legitimate alternative that the existing kernel already supports, which would make the 3D work practically unnecessary if 2D were competitive on accuracy for this task. Rather than argue the point, the 2D arm is included as the control condition and the comparison is measured directly.

## Feasibility Audit: Survives with Named Limitations

Based on verified resource facts and direct measurement, not duration estimates.

**Code and technique access.** The Im2win codebase is real, substantive, Apache-2.0 licensed, and verified to build and run on this project's exact environment (CUDA 12.8, sm_86, RTX 3090), reproducing the authors' kernel-level result. Its repository contains only forward-pass CUDA kernels, with no backward pass and no PyTorch or LibTorch operator binding anywhere; the `libtorch_tflops` directory links LibTorch only as a comparison target. The kernels are hardcoded to half precision with no fp32 path, and their tile and warp shapes are compile-time constants hand-tuned per problem shape.

**Baseline access.** nnU-Net's v1-lineage codebase, bundled inside MedNeXt's repository, is directly usable and was verified to build, run forward and backward passes, and profile correctly on the project's GPU. MedNeXt's repository is complete and usable but not under active development. EffiDec3D's repository is official and recent; its UT Austin Research License permits academic, research, and experimental use and permits modification, but Section 3.1(a) prohibits redistribution, so its code is cloned by the user rather than vendored into this repository.

**Data access.** The Medical Segmentation Decathlon is fully public with no registration or approval process. Verified directly: 131 labeled training volumes and 70 unlabeled test volumes for the Liver task, with three classes, downloaded, converted, and preprocessed to completion on the project's GPU instance.

**Compute access.** Verified by measurement on the project's rented RTX 3090: nnU-Net's 3D U-Net at the 128 cubed patch size runs at 314 ms per training iteration with mixed precision (702 ms without), using 5.07 GB and 9.65 GB peak respectively at batch size 2, and 55.5 ms per patch at inference. The GPU class matches the one used in Im2win's own published evaluation, which makes the reproduction directly comparable.

**Named limitation 1.** The windowing scheme has never been exercised on a genuinely three-dimensional kernel at any size. The open question is dimensionality rather than kernel size.

**Named limitation 2.** nnU-Net's 3D U-Net spans convolution channel widths from 32 to 640 (verified directly), a wider range than a single hand-tuned tile configuration is likely to cover well. Either a shape-adaptive design or several tuned configurations may be required, which is more implementation work than one fixed shape.

**Named limitation 3.** Profiling establishes a hard ceiling on any end-to-end claim. Convolution accounts for 65.5 percent of inference time and 55 percent of training time, so by Amdahl's law even an infinitely fast convolution kernel caps end-to-end inference improvement at 2.9 times. A kernel-level result must not be presented as if it transferred directly to end-to-end performance.

**Named limitation 4.** The backward kernels, if pursued, have no existing reference to validate against beyond gradient-checking against autograd on small synthetic inputs, and the weight-gradient pass has a different GEMM shape from the forward pass rather than mirroring it. This is the highest-risk element in the project and is therefore scoped as a secondary objective, outside the critical path for the primary inference result.

**Named limitation 5.** The fp16-only kernels interact with nnU-Net's InstanceNorm layers, which autocast keeps in fp32, inserting casts at every convolution boundary. This is a real overhead in the end-to-end measurement that does not exist in the isolated kernel benchmark, and the two numbers must be reported separately rather than conflated.
