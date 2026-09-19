"""Shape-aware dispatch between windowed convolution and cuDNN.

The crossover map (bench/bench_crossover.py) establishes that neither algorithm
dominates: measured as a geometric mean over the grid, always-cuDNN sits 1.135x
off a perfect oracle and always-windowed sits 1.164x off. Which one wins is a
function of layer shape, and it is predictable.

That makes the practical artifact a dispatcher rather than a replacement. This
module implements two policies:

  rule      decide from shape alone, at zero runtime cost
  autotune  time both once per distinct shape and cache the winner

The autotuner is the upper bound the rule is measured against. It is exact by
construction but pays a probing cost on first sight of each shape, and it can
only choose well for shapes it actually encounters.

The rule is deliberately two clauses, not a fitted surface. Section 6.5 of the
manuscript explains why a closed form is not justified here: the ratio is a
quotient of two shape-dependent efficiencies, and cuDNN's own efficiency varies
with shape, so a formula fitted to the windowed side alone would be overfitted
to one GPU. These two clauses are the part of the surface that held in every
configuration measured.
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from winconv import im2win_conv3d_minmat, im2win_conv3d_depthwise


def prefer_windowed(channels, spatial, kernel_size):
    """The decision rule, derived from the 60-configuration crossover grid.

    Returns True when the windowed path is expected to beat cuDNN.

    Clause 1 excludes kernel size 3, where windowed lost in all 12 shapes
    measured. Clause 2 admits wide layers at kernel size 5 and above, where it
    won in all 12.

    Clause 3 admits large kernels generally and is the weak one: it holds in
    18 of 24, not unanimously. The six exceptions are not scattered. Five sit
    at low channel count (30) or at the single stubborn 120-channel, 32-cubed
    family, always with spatial extent 32 or above, and every one of the four
    shapes at spatial extent 16 wins. So the failure mode is narrow layers at
    large spatial extent, which is the same low-intensity corner the rest of
    the paper identifies.

    We deliberately do NOT add a third clause to capture that. A guard on
    output shrinkage was tried and improved nothing at any threshold, and
    fitting another parameter to six data points would be overfitting to this
    grid rather than learning the surface. Clause 3 is the one to drop first
    when porting to different hardware, and bench_dispatch.py tests whether it
    survives on shapes it was never derived from.
    """
    if kernel_size <= 3:
        return False
    if channels >= 240 and kernel_size >= 5:
        return True
    return kernel_size >= 9


def _windowed(x, w, stride, depthwise):
    if depthwise:
        return im2win_conv3d_depthwise(x, w, stride=stride)
    return im2win_conv3d_minmat(x, w, stride=stride)


class DispatchingConv3d(nn.Module):
    """Drop-in nn.Conv3d that picks the faster convolution per layer shape.

    policy:
      'rule'     consult prefer_windowed() on the first forward pass
      'autotune' time both paths on the first forward pass and keep the winner
      'cudnn'    always cuDNN, the baseline
      'windowed' always windowed, the other baseline

    The choice is made once per distinct input shape and cached, so steady
    state costs one dictionary lookup.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, policy="rule"):
        super().__init__()
        # nn.Conv3d normalizes these to tuples; the windowed path wants scalars.
        # Comparing a tuple to an int is silently False, which disabled the
        # windowed path on every layer once already in this project.
        kernel_size = kernel_size[0] if isinstance(kernel_size, (tuple, list)) else kernel_size
        stride = stride[0] if isinstance(stride, (tuple, list)) else stride
        padding = padding[0] if isinstance(padding, (tuple, list)) else padding
        dilation = dilation[0] if isinstance(dilation, (tuple, list)) else dilation
        groups = groups[0] if isinstance(groups, (tuple, list)) else groups

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.policy = policy

        self.depthwise = groups > 1 and groups == in_channels == out_channels
        # shapes the windowed path cannot serve at all
        self.eligible = (dilation == 1 and kernel_size > 1
                         and (groups == 1 or self.depthwise))

        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups,
            kernel_size, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        self._choice = {}   # input shape -> bool, True means use windowed

    def _cudnn(self, x):
        return F.conv3d(x, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

    def _decide(self, x):
        """Choose a path for this input shape, once."""
        if not self.eligible:
            return False
        if self.policy == "cudnn":
            return False
        if self.policy == "windowed":
            return True
        if self.policy == "rule":
            return prefer_windowed(self.in_channels, x.shape[-1], self.kernel_size)
        if self.policy == "autotune":
            return self._time_both(x)
        raise ValueError(f"unknown policy {self.policy!r}")

    def _time_both(self, x, reps=5, iters=10, warmup=10):
        """Time both paths on the real input and return True if windowed wins.

        Timed under no_grad on a detached input, so autotuning never perturbs
        the forward pass it is called from.

        This is deliberately not a quick probe. It is the upper bound the rule
        is measured against, so if it is noisy the whole comparison is
        meaningless. Two specific hazards are handled:

        - cuDNN's own algorithm autotuner (cudnn.benchmark) needs several
          calls before it settles on an algorithm. Timing inside that window
          measures the search, not the kernel.
        - at small spatial extents the kernel is short enough that launch
          overhead and host jitter dominate a single reading.

        So each path gets a real warmup and the MEDIAN of several batches is
        taken, matching the discipline used throughout this project after
        single-shot readings produced three wrong headline numbers.
        """
        xd = x.detach()
        xp = F.pad(xd, (self.padding,) * 6) if self.padding > 0 else xd
        w = self.weight.detach()

        def run_win():
            _windowed(xp, w, self.stride, self.depthwise)

        def run_ref():
            self._cudnn(xd)

        med = {}
        with torch.no_grad():
            for name, fn in (("cudnn", run_ref), ("win", run_win)):
                for _ in range(warmup):
                    fn()
                torch.cuda.synchronize()
                samples = []
                for _ in range(reps):
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        fn()
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - t0) / iters)
                samples.sort()
                med[name] = samples[len(samples) // 2]
        self._last_timing = med
        return med["win"] < med["cudnn"]

    def forward(self, x):
        key = tuple(x.shape)
        use_win = self._choice.get(key)
        if use_win is None:
            use_win = self._decide(x)
            self._choice[key] = use_win
        if not use_win:
            return self._cudnn(x)
        xp = F.pad(x, (self.padding,) * 6) if self.padding > 0 else x
        o = _windowed(xp, self.weight, self.stride, self.depthwise)
        if self.bias is not None:
            o = o + self.bias.view(1, -1, 1, 1, 1)
        return o


def convert(model, policy="rule"):
    """Replace every nn.Conv3d in a model, preserving weights exactly.

    Returns (total replaced, number eligible for the windowed path).
    """
    total = eligible = 0
    for _, mod in model.named_modules():
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Conv3d):
                new = DispatchingConv3d(
                    child.in_channels, child.out_channels, child.kernel_size,
                    child.stride, child.padding, child.dilation, child.groups,
                    child.bias is not None, policy=policy,
                ).to(child.weight.device)
                new.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    new.bias.data = child.bias.data.clone()
                setattr(mod, name, new)
                total += 1
                eligible += int(new.eligible)
    return total, eligible
