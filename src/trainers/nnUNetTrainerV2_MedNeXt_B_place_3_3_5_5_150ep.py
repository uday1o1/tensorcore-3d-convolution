"""MedNeXt-B, kernel placement 3-3-5-5, 150 epochs, UpKern initialized.

One arm of the placement study. See mednext_placement_base for why UpKern
initialization is mandatory here rather than optional.
"""
import sys
from pathlib import Path

# nnU-Net imports trainers as modules of its own package, so a sibling file is
# not importable by bare name. Resolve this file (following the symlink the
# setup script installs) and put its real directory on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mednext_placement_base import MedNeXtPlacementTrainer  # noqa: E402


class nnUNetTrainerV2_MedNeXt_B_place_3_3_5_5_150ep(MedNeXtPlacementTrainer):
    KERNELS = [3, 3, 5, 5]
