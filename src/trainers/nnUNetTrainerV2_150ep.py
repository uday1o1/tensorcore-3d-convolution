from nnunet_mednext.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2

class nnUNetTrainerV2_150ep(nnUNetTrainerV2):
    '''Reduced schedule. The kernel comparison is performed on identical
    weights, so absolute Dice need not match published SOTA; what matters is
    that the model is realistically trained and that both kernels see the
    same weights.'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_num_epochs = 150
