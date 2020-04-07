from mnmt.inputter import ModuleArgsFeeder
from mnmt.inputter import TrainerArgsFeeder
import torch


class ArgsFeeder:

    def __init__(self,
                 encoder_args_feeder: ModuleArgsFeeder,
                 decoder_args_feeder: ModuleArgsFeeder,
                 trainer_args_feeder: TrainerArgsFeeder,
                 batch_size, src_pad_idx):
        """
        Args:
            encoder_args_feeder (ModuleArgsFeeder):
            decoder_args_feeder (ModuleArgsFeeder):
            trainer_args_feeder (TrainerArgsFeeder):
            batch_size: number of samples in a batch
        """
        self.encoder_args_feeder = encoder_args_feeder
        self.decoder_args_feeder = decoder_args_feeder
        self.trainer_args_feeder = trainer_args_feeder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("The current device for PyTorch is {}".format(self.device))
        self.batch_size = batch_size
        self.src_pad_idx = src_pad_idx





