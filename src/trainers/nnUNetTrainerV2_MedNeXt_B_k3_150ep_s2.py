"""MedNeXt-B kernel 3, 150 epochs, seed 2.

One of a pair of trainers differing only in random seed. They exist to measure
the run-to-run Dice spread of this training setup, which decides whether the
kernel-placement study is measurable at all.

The motivation is a number from MedNeXt's own paper: uniform kernel 5 with
UpKern initialization improves mean Dice by roughly 0.2 points over kernel 3
(84.03 to 84.23 on BTCV, 86.71 to 87.06 on AMOS22). A study that compares
kernel placements is therefore trying to resolve differences of that order. If
two identically configured runs of this setup differ by more than that, the
comparison cannot be made on a single run per configuration and needs either
repeats or a redesign. Running this first is cheaper than discovering it after
four full training runs.

nnU-Net's base trainer hardcodes seed 12345 when deterministic is set, so the
seed is reapplied here after the parent constructor. Network initialization
happens later, in initialize(), so both weight initialization and the data
augmentation stream differ between the two trainers.

These two runs are not overhead. Both are the 3-3-3-3 configuration, so one
serves as the study's baseline and as the UpKern source for the large-kernel
variants, and the pair gives the error bar.
"""
import numpy as np
import torch

from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import (
    nnUNetTrainerV2_MedNeXt_B_kernel3)

SEED = 2


class nnUNetTrainerV2_MedNeXt_B_k3_150ep_s2(nnUNetTrainerV2_MedNeXt_B_kernel3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_num_epochs = 150
        self.seed_used = SEED
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        self.print_to_log_file(f"variance pilot: reseeded to {SEED} "
                               f"(nnU-Net default is 12345)")
