from mnmt.inputter import ModuleArgsFeeder
import torch


class ArgsFeeder:

    def __init__(self,
                 encoder_args_feeder: ModuleArgsFeeder,
                 decoder_args_feeder: ModuleArgsFeeder,
                 batch_size: int, src_pad_idx: int, trg_pad_idx: int,
                 optim_choice: str, learning_rate: float, decay_patience: int,
                 lr_decay_factor: float, valid_criterion: str,
                 early_stopping_patience: int,
                 total_epochs: int, report_interval: int, exp_num: int,
                 multi_task_ratio,
                 data_container, src_lang, trg_lang, auxiliary_name=None):
        """
        Args:
            encoder_args_feeder (ModuleArgsFeeder):
            decoder_args_feeder (ModuleArgsFeeder):
            batch_size: number of samples in a batch
            src_pad_idx:
            trg_pad_idx:
            optim_choice: "Adam", "SGD", etc.
            learning_rate:
            decay_patience:
            lr_decay_factor:
            valid_criterion: "ACC" or "LOSS"
            early_stopping_patience:
            total_epochs:
            report_interval:
            exp_num:
            multi_task_ratio:
            data_container:
            src_lang:
            trg_lang:
            auxiliary_name:
        """
        self.encoder_args_feeder = encoder_args_feeder
        self.decoder_args_feeder = decoder_args_feeder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("The current device for PyTorch is {}".format(self.device))
        self.batch_size = batch_size
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        # For the optimizer
        self.learning_rate = learning_rate
        self.decay_patience = decay_patience
        self.optim_choice = optim_choice
        self.lr_decay_factor = lr_decay_factor
        self.valid_criterion = valid_criterion
        # For early stopping
        self.early_stopping_patience = early_stopping_patience
        # General settings
        self.total_epochs = total_epochs
        self.report_interval = report_interval
        self.exp_num = exp_num
        self.multi_task_ratio = multi_task_ratio
        self.data_container = data_container
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.auxiliary_name = auxiliary_name







