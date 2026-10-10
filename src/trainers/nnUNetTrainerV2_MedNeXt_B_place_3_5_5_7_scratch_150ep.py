"""MedNeXt-B, placement 3-5-5-7, 150 epochs, trained FROM SCRATCH.

The training strategy arm of the placement study, paired with the UpKern
initialized 3-5-5-7 run. MedNeXt reports that large kernel networks trained from
scratch fail to beat small kernels, and that UpKern is what makes them work. Its
large kernels sat in every stage, including the first at full resolution. Here
the first stage keeps kernel 3, so this arm asks whether UpKern is needed for
large kernels as such, or only for large kernels at high resolution.
"""
import sys
from pathlib import Path

# nnU-Net imports trainers as modules of its own package, so a sibling file is
# not importable by bare name. Resolve this file (following the symlink the
# setup script installs) and put its real directory on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mednext_placement_base import MedNeXtPlacementTrainer  # noqa: E402


class nnUNetTrainerV2_MedNeXt_B_place_3_5_5_7_scratch_150ep(MedNeXtPlacementTrainer):
    KERNELS = [3, 5, 5, 7]
    UPKERN = False
