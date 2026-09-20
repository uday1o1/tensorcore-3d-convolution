"""MedNeXt-B, kernel placement 3-3-5-5, 150 epochs, UpKern initialized.

One arm of the placement study. See mednext_placement_base for why UpKern
initialization is mandatory here rather than optional.
"""
from mednext_placement_base import MedNeXtPlacementTrainer


class nnUNetTrainerV2_MedNeXt_B_place_3_3_5_5_150ep(MedNeXtPlacementTrainer):
    KERNELS = [3, 3, 5, 5]
