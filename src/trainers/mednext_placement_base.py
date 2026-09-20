"""Base trainer for the kernel-placement study.

Subclasses set KERNELS, one entry per encoder stage. The network is built at
kernel 3, the depthwise convolutions are reassigned per stage, and the weights
are initialized from a trained kernel-3 checkpoint by UpKern.

Why UpKern rather than training from scratch: MedNeXt's paper reports that
large kernel MedNeXts trained from scratch fail to perform, scoring 84.03 on
BTCV against 84.23 with UpKern, and that without it large kernel performance is
indistinguishable from small kernels. Training these placements from scratch
would risk measuring an optimization failure and reporting it as evidence that
receptive field does not help, which is the wrong conclusion from the right
number.

The UpKern source is the kernel-3 run from the variance pilot, which is also
this study's baseline, so no training is spent solely on initialization.

Set MEDNEXT_UPKERN_SOURCE to the source checkpoint. The trainer refuses to run
without it rather than silently falling back to random initialization, since
that fallback would produce exactly the misleading null described above.
"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kernel_placement import set_stage_kernels, receptive_field

from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import (
    nnUNetTrainerV2_MedNeXt_B_kernel3)
from nnunet_mednext.run.load_weights import upkern_load_weights


class MedNeXtPlacementTrainer(nnUNetTrainerV2_MedNeXt_B_kernel3):
    """MedNeXt-B with per-stage depthwise kernels, UpKern initialized."""

    KERNELS = [3, 3, 3, 3]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_num_epochs = 150

    def initialize_network(self):
        # Builds MedNeXt-B at kernel 3.
        super().initialize_network()

        if list(self.KERNELS) == [3, 3, 3, 3]:
            self.print_to_log_file("placement 3-3-3-3, network left unchanged")
            return

        # Keep an untouched kernel-3 copy to serve as the UpKern source shape,
        # then retune the live network to the target placement.
        import copy
        reference = copy.deepcopy(self.network)

        changed = set_stage_kernels(self.network, self.KERNELS)
        if changed == 0:
            raise RuntimeError(
                f"no depthwise kernels changed for {self.KERNELS}; this network "
                "is identical to the baseline and the comparison would be empty")
        self.network.cuda()

        rf, counts = receptive_field(self.network, self.KERNELS)
        self.print_to_log_file(
            f"placement {'-'.join(map(str, self.KERNELS))}: retuned {changed} "
            f"depthwise layers, blocks per stage {counts}, receptive field {rf} voxels")

        src = os.environ.get("MEDNEXT_UPKERN_SOURCE")
        if not src or not Path(src).exists():
            raise RuntimeError(
                "MEDNEXT_UPKERN_SOURCE is unset or missing. Large kernel MedNeXt "
                "trained from scratch is reported by its authors to perform no "
                "better than small kernels, so a run without UpKern would measure "
                "an optimization failure rather than the effect of receptive "
                "field. Point it at the trained kernel-3 checkpoint.")

        state = torch.load(src, map_location="cpu", weights_only=False)
        reference.load_state_dict(state["state_dict"])
        self.network = upkern_load_weights(self.network, reference.cuda())
        self.print_to_log_file(f"UpKern initialized from {src}")
