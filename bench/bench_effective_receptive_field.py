"""Effective receptive field of each kernel placement, measured.

Section 7.1 ranks placements by theoretical receptive field, which counts how
far information could travel. It is an upper bound and it weights every path
equally. The effective receptive field is what the network actually uses: it is
smaller, roughly Gaussian, and it does not have to agree with the count.

There is a specific reason to expect disagreement here, and it runs against the
result the theoretical figure supports. Uniform 5-5-5-5 places a kernel 5 at
stage 0, where the jump is 1, so it widens the field directly at input
resolution. The 3-5-5-7 placement buys its field in deep stages, where each
contribution is spread across the upsampling path and may be diluted. If that
dilution is strong enough, the dominance of 3-5-5-7 over 5-5-5-5 could weaken
or invert, and that claim is load bearing for Part II.

Method, the standard one: run a forward pass on an input that requires
gradient, backpropagate from a single centre voxel of the output, and read the
spatial extent of nonzero input gradient. Reported two ways:

  support   the full extent of any nonzero gradient, comparable to the
            theoretical bound
  mass      the width containing a fixed fraction of total gradient magnitude,
            which is the honest measure of what the network actually weights

Weights are the initialization, not a trained network. This measures the
architecture's reach, not what training does with it, which is the right
quantity for a design question and the only one available given the accuracy
axis is not measurable here (Section 7.2).

Usage:  python bench_effective_receptive_field.py
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from kernel_placement import set_stage_kernels, receptive_field

PLACEMENTS = {
    "3-3-3-3": [3, 3, 3, 3],
    "3-3-5-5": [3, 3, 5, 5],
    "3-5-5-7": [3, 5, 5, 7],
    "5-5-5-5": [5, 5, 5, 5],
}
PATCH = 96          # large enough that the field is not clipped by the border
MASS_FRACTIONS = (0.5, 0.9)


def build(kernels):
    from nnunet_mednext.network_architecture.mednextv1.create_mednext_v1 import (
        create_mednextv1_base)
    model = create_mednextv1_base(num_input_channels=1, num_classes=3,
                                  kernel_size=3, ds=False).cuda()
    changed = set_stage_kernels(model, kernels)
    uniform_baseline = list(kernels) == [3, 3, 3, 3]
    if changed == 0 and not uniform_baseline:
        raise RuntimeError(f"no kernels changed for {kernels}")
    return model.eval(), changed


def gradient_profile(model, patch=PATCH):
    """1D profile of input gradient magnitude about the centre voxel."""
    x = torch.zeros(1, 1, patch, patch, patch, device="cuda", requires_grad=True)
    with torch.no_grad():
        x.normal_()
    x.requires_grad_(True)

    out = model(x)
    out = out[0] if isinstance(out, (list, tuple)) else out
    c = [d // 2 for d in out.shape[2:]]
    # Sum over channels at the centre voxel so the result is not specific to
    # one output class.
    out[0, :, c[0], c[1], c[2]].sum().backward()

    g = x.grad.detach().abs()[0, 0]
    # Collapse to one axis by summing the other two, giving the profile along z.
    prof = g.sum(dim=(1, 2))
    return prof.cpu(), g


def extent_from_profile(prof, fraction=None):
    """Width in voxels: full nonzero support, or the central mass fraction."""
    nz = (prof > 0).nonzero().flatten()
    if nz.numel() == 0:
        return 0
    if fraction is None:
        return int(nz[-1] - nz[0] + 1)
    total = prof.sum()
    if total <= 0:
        return 0
    centre = int(torch.argmax(prof))
    acc = prof[centre].clone()
    lo = hi = centre
    while acc < fraction * total and (lo > 0 or hi < len(prof) - 1):
        left = prof[lo - 1] if lo > 0 else torch.tensor(-1.0)
        right = prof[hi + 1] if hi < len(prof) - 1 else torch.tensor(-1.0)
        if right >= left:
            hi += 1
            acc = acc + prof[hi]
        else:
            lo -= 1
            acc = acc + prof[lo]
    return hi - lo + 1


def main():
    if not torch.cuda.is_available():
        print("needs a CUDA device")
        return 1
    torch.manual_seed(0)
    rows = []
    print(f"MedNeXt-B, patch {PATCH}^3, {torch.cuda.get_device_name(0)}")
    print("support is the full nonzero extent; mass50 and mass90 are the widths "
          "holding\nthat fraction of gradient magnitude\n")
    print(f"{'placement':<12}{'theoretical':>12}{'support':>9}{'mass90':>8}"
          f"{'mass50':>8}{'retuned':>9}")
    for name, kernels in PLACEMENTS.items():
        try:
            model, changed = build(kernels)
            theo, _ = receptive_field(model, kernels)
            prof, _ = gradient_profile(model)
            support = extent_from_profile(prof)
            m90 = extent_from_profile(prof, 0.9)
            m50 = extent_from_profile(prof, 0.5)
            rows.append({"placement": name, "kernels": kernels,
                         "theoretical": theo, "support": support,
                         "mass90": m90, "mass50": m50,
                         "layers_retuned": changed, "patch": PATCH,
                         "gpu": torch.cuda.get_device_name(0)})
            print(f"{name:<12}{theo:>12}{support:>9}{m90:>8}{m50:>8}{changed:>9}")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            torch.cuda.empty_cache()
            print(f"{name:<12}  failed: {type(e).__name__}: {e}")

    if not rows:
        return 1
    json.dump(rows, open(ROOT / "results" / "effective_receptive_field.json", "w"),
              indent=1)

    base = next(r for r in rows if r["placement"] == "3-3-3-3")
    print(f"\nrelative to {base['placement']}:")
    for r in rows:
        print(f"  {r['placement']:<10} theoretical "
              f"{r['theoretical']/base['theoretical']:5.2f}x   "
              f"support {r['support']/max(base['support'],1):5.2f}x   "
              f"mass90 {r['mass90']/max(base['mass90'],1):5.2f}x")

    # The claim under test: does 3-5-5-7 still beat 5-5-5-5 on reach?
    a = next((r for r in rows if r["placement"] == "3-5-5-7"), None)
    b = next((r for r in rows if r["placement"] == "5-5-5-5"), None)
    if a and b:
        print(f"\nPart II dominance claim, 3-5-5-7 against 5-5-5-5 "
              f"(3-5-5-7 costs 1.32x, 5-5-5-5 costs 2.00x):")
        for key in ("theoretical", "support", "mass90", "mass50"):
            holds = a[key] > b[key]
            print(f"  {key:<12} {a[key]:>5} vs {b[key]:>5}   "
                  f"{'HOLDS' if holds else 'DOES NOT HOLD'}")
        print("\n  Dominance requires more reach at lower cost. Cost is settled"
              "\n  by measurement in Section 7.1; if a reach column above does not"
              "\n  hold, the claim must be narrowed to the columns that do.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
