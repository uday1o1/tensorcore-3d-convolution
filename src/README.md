# Implementation

Shape-generic windowed convolution for two and three spatial dimensions, plus
the baselines it is compared against.

The published Im2win kernels this project reproduces are hand-specialized per
benchmark layer: their launch geometry contains literal constants (conv9 uses
`dim3(CEIL_DIV(M, TLIE_M_PER_BLOCK), 18, 54 * 6)`, where 54 is that layer's output height),
and at seven of the eight other shapes we tested they return silently zero
or partially computed output.
There is therefore no general implementation to port, and this code is a
from-scratch general implementation rather than an adaptation.

## Files

**`winconv.py`**: windowed convolution.

- `im2win_conv2d(x, w, stride)` materializes the H-axis overlap (expansion
  `k`, not `k^2`) and slides W inside the kernel. This mirrors the layout the
  released implementation produces, which is an H-unfold followed by a 1D
  convolution along W.
- `im2win_conv3d(x, w, stride)` unfolds D and H, slides W. Expansion `k^2`.
- `im2win_conv3d_minmat(x, w, stride)` unfolds D only, slides H and W.
  Expansion `k`. This is the variant the benchmarks, the drop-in module and the dispatcher
  use, since it matches
  the 2D method's materialization and makes the dimensional comparison
  like-for-like.
- `im2win_conv3d_depthwise(x, w, stride)` is the same minimal materialization
  for depthwise convolution (`groups == channels`), which is the layer type
  large-kernel 3D architectures actually use. It exists so the method can be
  measured on that layer type rather than assumed about. Depthwise reduces over
  one input channel instead of `C`, so materialization is unchanged while the
  compute it feeds drops by a factor of `C`, which the intensity account
  predicts is the worst case for any materializing method.
- `WindowedConv3d` is a drop-in replacement for `nn.Conv3d` supporting the
  padding, stride, and bias configurations nnU-Net uses. Serves dense and fully
  depthwise convolution; falls back to cuDNN for `kernel_size=1`, dilation
  above 1 and partially grouped convolution (`1 < groups < C`). No custom operator binding is needed
  because the implementation is expressed in PyTorch primitives.

The channel grouping in both 3D paths depends on the unfold ordering the
materialized axis as `(c, kd)` rather than `(kd, c)`. Getting this backwards is
silently wrong, not an error. `bench/verify_correctness.py` covers it on GPU;
the ordering is additionally validated in float64 against a naive reference by
`bench/verify_depthwise_indexing.py`, which needs only numpy.

**`fftconv.py`**: `fft_conv3d(x, w)`, FFT-based 3D convolution, stride 1.
Included as the natural third algorithm, since its cost depends on kernel size
only through padding. In frequency space the channel reduction becomes a dense
complex multiply whose cost the kernel-size savings do not touch.

**`dispatch.py`**: `prefer_windowed(channels, spatial, kernel_size)` is the
decision rule read off the crossover grid, and `DispatchingConv3d` is a drop-in
`nn.Conv3d` that applies it per layer shape and caches the choice. Policies are
`rule` (shape only, zero runtime cost), `autotune` (time both once per shape,
the measured reference the rule is compared with on whole networks), and the
two pure baselines.
`convert(model, policy)` swaps a whole network in place, preserving weights.

**`kernel_placement.py`**: per-stage kernel placement for MedNeXt.
`set_stage_kernels(model, kernels)` gives every depthwise convolution at each
resolution level its own kernel size and returns how many layers changed;
`receptive_field(model, kernels)` is the theoretical receptive field along one
axis, counting every depthwise convolution at the four highest-resolution
levels; `upkern_from_checkpoint` initializes a large-kernel network from a
trained kernel 3 one with MedNeXt's UpKern.

**`benchenv.py`**: shared benchmark plumbing. `results_dir()` is
`$BENCH_RESULTS_DIR` or `results/`; `device_tag()` names output files by GPU;
`device_meta()` records GPU, driver, power limit, clocks, host processor and
library versions for each row; `timed_pair` alternates which algorithm is
timed first.

**`trainers/`**: nnU-Net v1 trainers for the training study, installed into
the MedNeXt fork by `scripts/setup_vast.sh`: the kernel 3 seed pair
(`..._k3_150ep_s1`, `_s2`), the placements built on `mednext_placement_base.py`
(3-3-5-5, 3-5-5-7 and 5-5-5-5 with UpKern, 3-5-5-7 from scratch, and kernel 3
continued by the UpKern protocol), and nnU-Net at 150 epochs.

The accuracy comparison on identical trained weights lives in
`bench/bench_dice_identical_weights.py`, since it produces a measured result.

## Correctness

Run `bench/verify_correctness.py` before trusting any timing.

Two traps this code is built around:

1. **Silent wrongness.** The failure mode of the kernels being studied is
   returning zeros with no error. Verify output, not just runtime. A sweep
   with no output check reported 13,483 TFLOPS, roughly 95x the RTX 3090's
   142 TFLOPS dense FP16 Tensor Core peak, which is what timing an empty
   launch measures (`results/logs/measurements/sweep_unverified_nnunet_shapes.log`).
2. **TF32.** PyTorch enables it by default for cuDNN convolutions on Ampere and later. Its 10-bit mantissa
   produces relative errors of 2.65e-04 to 5.55e-04 here, which look like
   algorithmic bugs (`results/logs/measurements/winconv_correctness_tf32_on.log`).
   Disable it when verifying, or precision gets misattributed to correctness.

With TF32 disabled, every implementation here matches `F.conv2d` / `F.conv3d`
to a relative error between 0.00e+00 and 2.42e-06 in the committed correctness
logs (`results/logs/measurements/*correctness*.log`).

## Usage

```python
import sys; sys.path.insert(0, 'src')   # from the repository root
import torch
from winconv import im2win_conv3d_minmat, WindowedConv3d

x = torch.randn(2, 60, 64, 64, 64, device='cuda', dtype=torch.half)
w = torch.randn(60, 60, 9, 9, 9, device='cuda', dtype=torch.half)
y = im2win_conv3d_minmat(x, w)          # 1.20x cuDNN on the RTX 3090, 0.94x on the RTX 4090

layer = WindowedConv3d(60, 60, 3, padding=1).cuda()   # drop-in for nn.Conv3d
```

## When to use this

At kernel size 3 the windowed method loses to cuDNN on the median in all 12 grid shapes
on the RTX 3090 (and in 7 of 8 held-out shapes, the eighth at 1.009), by a factor of 1.1 to 1.9, and substituted into a whole network it is 2.34x
slower on 3.9x the memory. At 240 channels with kernel size 5 or larger it wins
in 11 of the 12 shapes measured, the twelfth at parity (0.997), the largest margin 1.68x. The largest speedup across the whole grid
is 2.36x. See `bench/bench_crossover.py` for the map and its trial spread.
