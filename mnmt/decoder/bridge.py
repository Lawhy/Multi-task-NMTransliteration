import torch
import torch.nn as nn


class BridgeLayer(nn.Module):
    """extra bridge layer for encoder final hidden to be transformed into
    decoder's initial hidden
    """
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, num_of_states):
        """
        Args:
            encoder_hidden_dim:
            decoder_hidden_dim:
            num_of_states:
        """
        super().__init__()
        self.num_of_states = num_of_states
        in_dim = encoder_hidden_dim * 2
        out_dim = decoder_hidden_dim
        self.bridge_layer = nn.ModuleList(nn.Linear(in_dim, out_dim) for _ in range(self.num_of_states))

    def forward(self, encoder_final_state):
        """
        Args:
            encoder_final_state: [batch_size, encoder_hidden_dim * 2], tuple for LSTM
        """
        if self.num_of_states == 1:
            initial_decoder_hidden = torch.tanh(self.bridge_layer[0](encoder_final_state))
        else:
            initial_decoder_hidden = (torch.tanh(self.bridge_layer[0](encoder_final_state[0])),
                                      torch.tanh(self.bridge_layer[1](encoder_final_state[1])))
        return initial_decoder_hidden
