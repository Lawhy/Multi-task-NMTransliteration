from mnmt.inputter import ArgsFeeder
from mnmt.encoder import BasicEncoder
from mnmt.trainer.utils import create_mask
import torch.nn as nn


class Seq2MultiSeq(nn.Module):

    def __init__(self, args_feeder: ArgsFeeder, encoder: BasicEncoder,
                 decoder_list: nn.ModuleList, teacher_forcing_ratio):
        super().__init__()
        self.args_feeder = args_feeder
        self.encoder = encoder
        self.decoder_list = decoder_list
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device = self.args_feeder.device

    def forward(self, src, src_lens, *trg):
        encoder_outputs, encoder_final_state = self.encoder(src, src_lens)
        mask = create_mask(src, self.args_feeder.src_pad_idx)
        output_pred = []
        for i in range(len(self.decoder_list)):
            output, pred = self.decoder_list[i](trg[i], encoder_outputs, encoder_final_state, mask,
                                                self.teacher_forcing_ratio)
            output_pred += [output, pred]
        # output1, pred1, output2, pred2 ...
        return tuple(output_pred)
