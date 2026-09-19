from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import nnUNetTrainerV2_MedNeXt_B_kernel3

class nnUNetTrainerV2_MedNeXt_B_kernel3_150ep(nnUNetTrainerV2_MedNeXt_B_kernel3):
    '''MedNeXt-B k3 on a 150 epoch schedule, matched to the nnU-Net baseline
    so the two are directly comparable.'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_num_epochs = 150
