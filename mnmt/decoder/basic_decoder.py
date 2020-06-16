import torch
import torch.nn as nn


class BasicDecoder(nn.Module):

    def __init__(self, feed_forward_decoder, bridge_layer, device):
        """
        Args:
            feed_forward_decoder:
            bridge_layer:
        """
        super().__init__()
        self.feed_forward_decoder = feed_forward_decoder
        self.bridge_layer = bridge_layer
        self.trg_vocab_size = self.feed_forward_decoder.attrs.input_dim
        self.device = device

    def init_s_0(self, encoder_final_state):
        """
        Args:
            encoder_final_state:  [batch_size, encoder_hidden_dim * 2]
        """
        return self.bridge_layer(encoder_final_state)  # [batch_size, hidden_dim]

    def init_decoder_outputs(self, max_length, batch_size):
        init_outputs = torch.zeros(max_length, batch_size, self.trg_vocab_size).to(self.device)
        return init_outputs

    def forward(self, trg, encoder_outputs, encoder_final_state, mask, teacher_forcing_ratio):
        """
        Args:
            trg:  [trg_length, batch_size], target samples batch
            encoder_outputs:  [src_length, batch_size, encoder_hidden_dim * 2]
            encoder_final_state:  [batch_size, encoder_hidden_dim * 2]
            mask: [batch_size, src_length], mask out <pad> for attention
            teacher_forcing_ratio: probability of applying teacher forcing or not
        """
        raise NotImplementedError
