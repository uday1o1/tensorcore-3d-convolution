# Part I measurement logs

Verbatim records of the measurements behind the Im2win reproduction, the
generality audit, correctness, dimensional scaling, FFT, profiling and the
whole-network substitution. All were taken on 2026-09-19 on the RTX 3090 host
recorded in `host_rtx3090_main_versions.log` (driver 595.71.05, PyTorch
2.11.0+cu128, CUDA 12.8, cuDNN 9.19.0).

Each file has three blocks: `# command` (the exact command, with the host
address replaced by `<host>`), `# date` (UTC) and `# output` (verbatim, with
the provider's login banner removed). Most commands are short inline scripts;
the scripts in `bench/` are their consolidated form.

| Log | Records |
|---|---|
| `im2win_conv9_authors_harness_50it.log` | Im2win's released Tensor Core kernel on conv9 in the authors' harness, best of 50: 37087.4 GFLOPS |
| `authors_harness_source_50it.log` | the authors' harness source: 50 timed iterations, window transform outside the timing loop |
| `sweep_harness_source_30it.log` | the command that wrote the base sweep harness (30 timed iterations); `bench/sweep_convs.cpp` is that harness without its local `tensorDimensions` struct (Im2win's `Im2win_bchw.hpp` defines it), with the output check of `sweep_nonzero_check.log` and the mean of `im2win_conv9_best_mean_30it.log` added |
| `im2win_conv9_best_mean_30it.log` | conv9 in the sweep harness: 37.0375 TFLOPS best, 36.4182 mean, output fully populated |
| `cudnn_conv9_events_best_of_50.log` | cuDNN on conv9, CUDA events, best of 50: 2.314 ms, 23.8 TFLOPS; kernel-only ratio 1.56 |
| `cudnn_conv9_three_methods.log` | cuDNN on conv9 by host clock, CUDA events and back to back: 22.96, 23.78, 23.24 TFLOPS |
| `cudnn_conv9_fresh_process.log` | cuDNN on conv9 in fresh processes: 17.62, 21.22, 21.07 TFLOPS |
| `im2win_gpu_transform_end_to_end.log` | window transform on the GPU, 0.561 ms; end to end ratio 1.13 |
| `sweep_unverified_nnunet_shapes.log` | throughput reported before output was checked, 13483.6 TFLOPS on an nnU-Net shape |
| `sweep_nonzero_check.log` | conv9 and the first nnU-Net shape with an output check: conv9 fully populated, the nnU-Net shape 0 of 124848000 outputs nonzero |
| `generality_audit.log` | conv9 with batch, channels and extent varied one at a time: fraction of output computed |
| `launch_geometry_conv9.log`, `launch_geometry_variants.log` | literal launch geometry in the released conv9 and conv4 kernels |
| `winconv_correctness_tf32_on.log`, `winconv_correctness_tf32_off.log` | windowed convolution against `F.conv2d`/`F.conv3d` with TF32 on (failures at 2.65e-04 to 5.55e-04) and off (0.00e+00 to 2.42e-06) |
| `minmat_correctness.log` | minimal-materialization 3D path against `F.conv3d` |
| `kernel_size_correctness_and_ratio.log` | correctness and cuDNN ratio across kernel sizes 3, 5, 7 |
| `verify_correctness_full.log` | `bench/verify_correctness.py` over the full set of implementations, FFT included |
| `fft_correctness_first.log` | FFT convolution against `F.conv3d`: 4.28e-07 to 1.37e-06 |
| `fft_landscape.log` | FFT convolution against cuDNN at 60 and 120 channels: 0.01 to 0.35 |
| `generic_windowed_conv9_anchor.log` | generic windowed implementation on conv9: 12.87 TFLOPS compute phase, 2.88x below the hand-tuned kernel |
| `dimscale_2d_vs_3d.log` | nnU-Net 2D and 3D layer shapes, generic windowed against cuDNN: 2D geometric mean 0.591 |
| `dimscale_materialization.log` | 3D with `k` against `k^2` materialization: 0.654 and 0.429, per layer 0.22 to 0.96 |
| `nnunet_flops_conv_share.log` | nnU-Net 3D at base width 30: 422.1 G multiply-accumulates (fvcore's FLOP count), 99.6 percent in convolution including transposed convolution |
| `mednext_conv1_flops_first.log`, `mednext_conv1_and_nnunet_profile.log` | MedNeXt depthwise `conv1` share of FLOPs, 6.2 percent; nnU-Net inference profile, convolution 65.6, normalization 13.6, layout conversion and memory copies 12.2 percent |
| `layout_conversion_shares.log` | layout conversion and memory copy share of GPU time for one forward convolution at three layer shapes, cuDNN against windowed |
| `nnunet_plans_patch_batch.log` | nnU-Net plans for MSD Liver: 3D batch 2, patch 128 cubed; 2D batch 16, patch 512 |
| `nnunet_preprocessed_131_volumes.log` | 131 preprocessed training volumes |
| `end_to_end_substitution.log` | windowed substitution into nnU-Net 3D on identical weights: 27 `Conv3d` layers, 22 windowed; rel_err 1.18e-06; cuDNN 53.4 ms and 1.31 GB against windowed 125.0 ms and 5.07 GB |
| `dice_compare_script_checkpoint.log` | the command that wrote the identical-weights Dice script, including the checkpoint it loads (`nnUNetTrainerV2_150ep`, fold 0, `model_best.model`) |
| `dice_identical_weights_27_cases.log` | Dice on 27 validation cases with cuDNN and with the windowed path, same weights; `../../dice_identical_weights.json` holds the per-case values |
| `host_rtx3090_main_versions.log` | GPU, driver, PyTorch, CUDA and cuDNN versions of the host |
