"""Per-stage kernel placement for MedNeXt, and UpKern initialization.

MedNeXt exposes one kernel size for the whole network. The question under study
is where a large kernel is worth spending, so the depthwise convolutions are
reassigned per stage after construction.

Two facts from MedNeXt's own paper shape this module:

  1. Large kernel MedNeXts trained from scratch fail to perform. Their kernel 5
     model scores 84.03 on BTCV from scratch against 84.23 with UpKern
     initialization, and without UpKern large kernel performance is
     indistinguishable from small kernels. Any placement study that trains from
     scratch therefore risks measuring an optimization failure and reporting it
     as "receptive field does not help".
  2. The benefit being chased is roughly 0.2 Dice. Comparisons at that scale
     only mean something once run-to-run spread is known, which is what the
     variance pilot measures.

This module is shared by the benchmark and the trainers so that the network
being timed and the network being trained are built by the same code.
"""
import torch
import torch.nn as nn


def depthwise_convs(model):
    """Every depthwise convolution in the model, in module order."""
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Conv3d) and m.groups == m.in_channels
            and m.in_channels > 1]


def set_stage_kernels(model, kernels):
    """Give each encoder stage its own depthwise kernel size.

    A Conv3d's kernel size cannot be reassigned in place because the weight
    tensor shape is fixed, so each is replaced by a new convolution with
    matching channels, groups and stride, and same padding.

    Stages are identified by channel count rather than by module path, since
    MedNeXt's attribute names are not a public interface. The 1x1x1 expansion
    and compression convolutions are untouched, which keeps this an experiment
    about receptive field rather than about width.

    Returns the number of layers changed. The caller must check it: a silent
    zero means every configuration is the same network, which would produce a
    comparison that was never actually made.
    """
    dw = depthwise_convs(model)
    if not dw:
        raise RuntimeError("no depthwise convolutions found; MedNeXt layout changed")
    widths = sorted({m.in_channels for _, m in dw})
    changed = 0
    for name, mod in dw:
        stage = min(widths.index(mod.in_channels), len(kernels) - 1)
        k = kernels[stage]
        if mod.kernel_size[0] == k:
            continue
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        new = nn.Conv3d(mod.in_channels, mod.out_channels, k,
                        stride=mod.stride, padding=k // 2,
                        groups=mod.groups, bias=mod.bias is not None)
        setattr(parent, parts[-1], new.to(mod.weight.device))
        changed += 1
    return changed


def blocks_per_stage(model, n_stages):
    """How many depthwise convolutions sit in each stage, widest last."""
    per = {}
    for _, m in depthwise_convs(model):
        per.setdefault(m.in_channels, []).append(m.kernel_size[0])
    widths = sorted(per)
    counts = [len(per[w]) for w in widths[:n_stages]]
    while len(counts) < n_stages:
        counts.append(1)
    return counts


def receptive_field(model, kernels, strides=(2, 2, 2, 1)):
    """Theoretical receptive field in input voxels along one spatial axis.

    Block counts come from the model. An earlier version assumed one
    convolution per stage and understated the result by about half, since
    MedNeXt-B carries two. The ratios between placements are nearly invariant
    to that count, so the ordering survived, but the absolute values did not.

    This is an upper bound. The effective receptive field is smaller and
    roughly Gaussian, and measuring it means backpropagating from a centre
    voxel and reading the extent of nonzero input gradient.
    """
    counts = blocks_per_stage(model, len(kernels))
    r, jump = 1, 1
    for k, n, st in zip(kernels, counts, strides):
        for _ in range(n):
            r += (k - 1) * jump
        jump *= st
    return r, counts


def upkern_from_checkpoint(target, reference_builder, checkpoint_path):
    """Initialize a large kernel network from a trained small kernel one.

    MedNeXt's own routine trilinearly interpolates weights whose spatial dims
    differ and copies everything else unchanged. It needs a module, not a state
    dict, so the reference network is built and loaded first.

    Returns the target network, initialized in place.
    """
    from nnunet_mednext.run.load_weights import upkern_load_weights

    reference = reference_builder()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reference.load_state_dict(state["state_dict"])
    return upkern_load_weights(target, reference)
