from nnunet_mednext.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2

class nnUNetTrainerV2_150ep(nnUNetTrainerV2):
    '''nnU-Net v1 on a 150 epoch schedule, the same as every MedNeXt trainer
    here. It is the nnU-Net row of the training study and the network for the
    identical-weights accuracy check, where windowed and cuDNN convolution run
    on these same trained weights, so absolute Dice need not match published
    results.'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_num_epochs = 150
