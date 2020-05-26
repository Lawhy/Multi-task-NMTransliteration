from mnmt.inputter import ArgsFeeder
from mnmt.encoder import BasicEncoder
from mnmt.decoder import BasicDecoder
from mnmt.trainer.utils import create_mask
import torch.nn as nn


class Seq2Seq(nn.Module):

    def __init__(self, args_feeder: ArgsFeeder, encoder: BasicEncoder, decoder: BasicDecoder, teacher_forcing_ratio):
        super().__init__()
        self.args_feeder = args_feeder
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device = self.decoder.device

    def forward(self, src, src_lens, trg):
        encoder_outputs, encoder_final_state = self.encoder(src, src_lens)
        mask = create_mask(src, self.args_feeder.src_pad_idx)
        prediction = self.decoder(trg, encoder_outputs, encoder_final_state, mask, self.teacher_forcing_ratio)
        return prediction
