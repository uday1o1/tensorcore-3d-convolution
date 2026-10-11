"""MedNeXt-B, placement 3-3-3-3, 150 epochs, continued from the kernel 3 run.

The control for UpKern. An UpKern placement starts from the trained kernel 3
network with a fresh optimizer and learning rate schedule, so it is trained
for 300 epochs in total against the baseline's 150. 3-3-5-5 and 3-5-5-7 with
UpKern both beat the baseline by about 0.001 liver Dice while 3-5-5-7 from
scratch did not, which is equally consistent with larger kernels helping and
with the extra training helping. This run applies the same protocol with no
kernel enlarged. A placement's gain counts towards receptive field only to the
extent it exceeds this run's.
"""
import sys
from pathlib import Path

# nnU-Net imports trainers as modules of its own package, so a sibling file is
# not importable by bare name. Resolve this file (following the symlink the
# setup script installs) and put its real directory on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mednext_placement_base import MedNeXtPlacementTrainer  # noqa: E402


class nnUNetTrainerV2_MedNeXt_B_place_3_3_3_3_continued_150ep(MedNeXtPlacementTrainer):
    KERNELS = [3, 3, 3, 3]
    CONTINUE = True
