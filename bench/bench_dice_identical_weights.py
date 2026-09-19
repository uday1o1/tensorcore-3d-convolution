"""Accuracy non-inferiority: cuDNN against windowed on IDENTICAL trained weights.

Both configurations run the same trained network over the same validation cases
with the same sliding window procedure, differing only in which convolution
kernel computes each layer. Because the weights are identical, any difference
is the kernel's numerics alone, with no training variance to confound it.

This is the one script in bench/ that needs artifacts beyond the repo: a
preprocessed nnU-Net dataset and a trained checkpoint. It locates them from
nnU-Net's own environment variables, or from explicit arguments, and tells you
precisely what is missing rather than failing on an obscure path error.

Prerequisites:
  - the MedNeXt fork installed (provides nnunet_mednext)
  - a preprocessed task directory, normally $nnUNet_preprocessed/<task>
  - a trained checkpoint, normally under $RESULTS_FOLDER

Usage:
  python bench_dice_identical_weights.py                       # env vars
  python bench_dice_identical_weights.py --cases 5             # quick check
  python bench_dice_identical_weights.py --preprocessed DIR --checkpoint FILE
"""
import argparse
import copy
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Resolve paths from this file, not the working directory, so the script
# runs from anywhere in a fresh clone.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from winconv import WindowedConv3d

torch.backends.cudnn.benchmark = True

TASK = "Task003_Liver"
STAGE = "nnUNetData_plans_v2.1_trgSp_1x1x1_stage1"
TRAINER = "nnUNetTrainerV2_150ep__nnUNetPlansv2.1_trgSp_1x1x1"
PATCH = (128, 128, 128)
BASE_FEATURES = 30   # the planner's value for this task, not the 32 default
NUM_CLASSES = 3      # background, liver, cancer


def build():
    from nnunet_mednext.network_architecture.generic_UNet import Generic_UNet
    return Generic_UNet(
        1, BASE_FEATURES, NUM_CLASSES, 5, 2, 2,
        nn.Conv3d, nn.InstanceNorm3d, {"eps": 1e-5, "affine": True},
        nn.Dropout3d, {"p": 0, "inplace": True},
        nn.LeakyReLU, {"negative_slope": 1e-2, "inplace": True},
        False, False, lambda x: x, None,
        [[2, 2, 2]] * 5, [[3, 3, 3]] * 6, False, True, True,
    ).cuda().eval()


def swap(model):
    """Replace every Conv3d with WindowedConv3d, preserving weights exactly."""
    active = 0
    for _, mod in model.named_modules():
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Conv3d):
                new = WindowedConv3d(
                    child.in_channels, child.out_channels, child.kernel_size,
                    child.stride, child.padding, child.dilation, child.groups,
                    child.bias is not None).cuda()
                new.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    new.bias.data = child.bias.data.clone()
                setattr(mod, name, new)
                active += int(new.use_windowed)
    return active


@torch.no_grad()
def sliding(model, vol):
    """Identical tiling procedure for both kernels, with edge clamp."""
    C, D, H, W = vol.shape
    pd, ph, pw = PATCH
    D2, H2, W2 = max(D, pd), max(H, ph), max(W, pw)
    x = torch.nn.functional.pad(vol.unsqueeze(0), (0, W2 - W, 0, H2 - H, 0, D2 - D))
    out = torch.zeros((1, NUM_CLASSES, D2, H2, W2), device="cuda", dtype=torch.float32)
    cnt = torch.zeros((1, 1, D2, H2, W2), device="cuda", dtype=torch.float32)
    step = [p // 2 for p in PATCH]
    zs = list(range(0, max(D2 - pd, 0) + 1, step[0])) or [0]
    ys = list(range(0, max(H2 - ph, 0) + 1, step[1])) or [0]
    xs = list(range(0, max(W2 - pw, 0) + 1, step[2])) or [0]
    for axis, size, patch in ((zs, D2, pd), (ys, H2, ph), (xs, W2, pw)):
        if axis[-1] != size - patch:
            axis.append(size - patch)
    for z in zs:
        for y in ys:
            for xx in xs:
                p = x[:, :, z:z + pd, y:y + ph, xx:xx + pw].cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    pred = model(p)
                out[:, :, z:z + pd, y:y + ph, xx:xx + pw] += pred.float()
                cnt[:, :, z:z + pd, y:y + ph, xx:xx + pw] += 1
    out = out / cnt.clamp(min=1)
    return out[:, :, :D, :H, :W].argmax(1)[0]


def dice(pred, gt, cls):
    p, g = (pred == cls), (gt == cls)
    total = p.sum().item() + g.sum().item()
    return (2 * (p & g).sum().item() / total) if total > 0 else float("nan")


def resolve(args):
    """Find the dataset and checkpoint, or explain exactly what is missing."""
    missing = []
    pre = args.preprocessed
    if pre is None:
        root = os.environ.get("nnUNet_preprocessed")
        if root:
            pre = Path(root) / TASK / STAGE
        else:
            missing.append(
                "preprocessed data: set $nnUNet_preprocessed, or pass "
                "--preprocessed <dir containing the .npz cases>")
    pre = Path(pre) if pre else None

    ck = args.checkpoint
    if ck is None:
        root = os.environ.get("RESULTS_FOLDER")
        if root:
            ck = (Path(root) / "nnUNet" / "3d_fullres" / TASK / TRAINER
                  / "fold_0" / "model_best.model")
        else:
            missing.append(
                "trained checkpoint: set $RESULTS_FOLDER, or pass "
                "--checkpoint <path to model_best.model>")
    ck = Path(ck) if ck else None

    splits = args.splits
    if splits is None and os.environ.get("nnUNet_preprocessed"):
        splits = Path(os.environ["nnUNet_preprocessed"]) / TASK / "splits_final.pkl"
    elif splits is None and pre is not None:
        splits = pre.parent / "splits_final.pkl"
    splits = Path(splits) if splits else None

    for label, path in (("preprocessed dir", pre), ("checkpoint", ck),
                        ("splits_final.pkl", splits)):
        if path is not None and not path.exists():
            missing.append(f"{label} does not exist: {path}")

    if missing:
        print("Cannot run. This benchmark needs a preprocessed dataset and a "
              "trained checkpoint, which are not in the repository:\n")
        for m in missing:
            print(f"  - {m}")
        print("\nEvery other benchmark in bench/ runs on synthetic tensors and "
              "needs only a GPU. See the README for which results need what.")
        return None
    return pre, ck, splits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preprocessed", help="directory holding the .npz cases")
    ap.add_argument("--checkpoint", help="trained model_best.model")
    ap.add_argument("--splits", help="splits_final.pkl defining the validation fold")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cases", type=int, default=0,
                    help="limit number of validation cases, 0 means all")
    ap.add_argument("--out", default=str(ROOT / "results" / "dice_identical_weights.json"))
    args = ap.parse_args()

    found = resolve(args)
    if found is None:
        return 1
    pre, ck_path, splits_path = found

    split = pickle.load(open(splits_path, "rb"))
    val = [str(v) for v in split[args.fold]["val"]]
    state = torch.load(ck_path, map_location="cpu", weights_only=False)["state_dict"]

    ref = build()
    ref.load_state_dict(state)
    win = copy.deepcopy(ref)
    active = swap(win)
    print(f"windowed path active on {active} layers")
    if active == 0:
        print("ERROR: windowed path never active, this would compare cuDNN with itself")
        return 1

    n = args.cases or len(val)
    rows = []
    for i, case in enumerate(val[:n]):
        d = np.load(str(pre / f"{case}.npz"))["data"]
        vol = torch.from_numpy(d[:-1]).float().cuda()
        gt = torch.from_numpy(d[-1]).cuda()
        pr, pw = sliding(ref, vol), sliding(win, vol)
        agree = (pr == pw).float().mean().item()
        r = {"case": case,
             "cudnn_liver": dice(pr, gt, 1), "cudnn_tumor": dice(pr, gt, 2),
             "win_liver": dice(pw, gt, 1), "win_tumor": dice(pw, gt, 2),
             "voxel_agreement": agree}
        rows.append(r)
        print(f"{i+1}/{n} {case}: cuDNN L{r['cudnn_liver']:.4f} T{r['cudnn_tumor']:.4f}"
              f" | win L{r['win_liver']:.4f} T{r['win_tumor']:.4f}"
              f" | agree {agree:.6f}", flush=True)
        del vol, gt, pr, pw
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=1)

    def mean(key):
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"\nmean liver Dice: cuDNN {mean('cudnn_liver'):.5f}  "
          f"windowed {mean('win_liver'):.5f}  "
          f"delta {mean('win_liver') - mean('cudnn_liver'):+.1e}")
    print(f"mean tumor Dice: cuDNN {mean('cudnn_tumor'):.5f}  "
          f"windowed {mean('win_tumor'):.5f}  "
          f"delta {mean('win_tumor') - mean('cudnn_tumor'):+.1e}")
    print(f"mean voxel agreement: {mean('voxel_agreement'):.6f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
