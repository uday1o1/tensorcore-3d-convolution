# Implementation

Shape-generic windowed convolution for two and three spatial dimensions, plus
the baselines it is compared against.

The published Im2win kernels this project reproduces are hand-specialized per
benchmark layer: their launch geometry contains literal constants (conv9 uses
`dim3(CEIL_DIV(M, TILE), 18, 54*6)`, where 54 is that layer's output height),
and they return silently zero or partially computed output at any other shape.
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
  Expansion `k`. This is the variant the paper's results use, since it matches
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
  depthwise convolution; falls back to cuDNN for `kernel_size=1` and partially
  grouped convolution (`1 < groups < C`). No custom operator binding is needed
  because the implementation is expressed in PyTorch primitives.

The channel grouping in both 3D paths depends on the unfold ordering the
materialized axis as `(c, kd)` rather than `(kd, c)`. Getting this backwards is
silently wrong, not an error, and it has already happened once in this project.
`bench/verify_correctness.py` covers it on GPU; the ordering is additionally
validated against a naive reference in exact arithmetic.

**`fftconv.py`**: `fft_conv3d(x, w)`, FFT-based 3D convolution, stride 1.
Included as the natural third algorithm, since its cost is independent of
kernel size. Measured here it is not competitive at realistic channel widths:
in frequency space the channel reduction becomes a dense complex multiply
whose cost the kernel-size savings do not touch.

**`dispatch.py`**: `prefer_windowed(channels, spatial, kernel_size)` is the
decision rule read off the crossover grid, and `DispatchingConv3d` is a drop-in
`nn.Conv3d` that applies it per layer shape and caches the choice. Policies are
`rule` (shape only, zero runtime cost), `autotune` (time both once per shape,
the exact upper bound the rule is scored against), and the two pure baselines.
`convert(model, policy)` swaps a whole network in place, preserving weights.

The accuracy comparison on identical trained weights lives in
`bench/bench_dice_identical_weights.py`, since it produces a measured result.

## Correctness

Run `bench/verify_correctness.py` before trusting any timing.

Two traps this code is built around:

1. **Silent wrongness.** The failure mode of the kernels being studied is
   returning zeros with no error. Verify output, not just runtime. An early
   sweep here reported 13,483 TFLOPS, roughly 190x the device peak, which is
   what timing an empty launch measures.
2. **TF32.** PyTorch enables it by default on Ampere. Its 10-bit mantissa
   produces relative errors near 3e-04 that look like algorithmic bugs.
   Disable it when verifying, or precision gets misattributed to correctness.

With TF32 disabled, every implementation here matches `F.conv2d` / `F.conv3d`
to between 0.00e+00 and 2.4e-06.

## Usage

```python
import torch
from winconv import im2win_conv3d_minmat, WindowedConv3d

x = torch.randn(2, 60, 64, 64, 64, device='cuda', dtype=torch.half)
w = torch.randn(60, 60, 9, 9, 9, device='cuda', dtype=torch.half)
y = im2win_conv3d_minmat(x, w)          # beats cuDNN at this shape

layer = WindowedConv3d(60, 60, 3, padding=1).cuda()   # drop-in for nn.Conv3d
```

## When to use this

At kernel size 3 the windowed method loses to cuDNN in all 12 shapes measured,
by a factor of two to three, and substituted into a whole network it is 2.34x
slower on 3.9x the memory. At 240 channels with kernel size 5 or larger it wins
in all 12 shapes measured, by up to 1.73x. Speedups across the whole grid reach
2.36x. See `bench/bench_crossover.py` for the map and its trial spread.
