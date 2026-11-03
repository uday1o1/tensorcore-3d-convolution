from nnunet_mednext.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2

class nnUNetTrainerV2_150ep(nnUNetTrainerV2):
    '''nnU-Net v1 on a 150 epoch schedule (nnU-Net's default is 1000), the
    same as every MedNeXt trainer here. Two separate trainings use it: the
    network for the identical-weights accuracy check, where windowed and cuDNN
    convolution run on the same trained weights (RTX 3090 benchmark host), and
    the nnU-Net row of the training study (RTX 4090 training host). Absolute
    Dice need not match published results.'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_num_epochs = 150
