"""MedNeXt-B kernel 3, 150 epochs, seed 2.

One of a pair of trainers (seeds 1 and 2) differing only in random seed.
Together they measure the run-to-run Dice spread of this training setup, the
noise floor against which kernel placements are compared.

The scale that floor must resolve comes from MedNeXt's own paper: uniform
kernel 5 with UpKern initialization improves Dice by roughly 0.2 points over
kernel 3 (84.01 to 84.23 on BTCV, 89.14 to 89.38 on AMOS22). If two identically
configured runs differ by more than that, one run per configuration cannot
resolve the comparison.

nnU-Net's base trainer seeds numpy and torch with 12345 only when deterministic
is set, which it is not by default, so the seed is applied here after the
parent constructor. Network initialization happens later, in initialize(), so
this seed fixes the weight initialization. The data augmentation workers seed
themselves from the OS (batchgenerators' MultiThreadedAugmenter with
seeds=None), so the augmentation stream differs between any two runs
regardless of this seed.

Both trainers are the 3-3-3-3 configuration. The seed 1 trainer is also the
placement study's baseline: a separate run of it supplies the baseline scores
and the UpKern source checkpoint for the large-kernel variants.
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
        # This text reproduces the committed training logs verbatim. Its
        # parenthesis is loose: nnU-Net seeds the global numpy and torch
        # generators with 12345 only when run with --deterministic.
        self.print_to_log_file(f"variance pilot: reseeded to {SEED} "
                               f"(nnU-Net default is 12345)")
