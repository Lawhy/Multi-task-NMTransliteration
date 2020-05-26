import torch
import torch.nn as nn


class Attention(nn.Module):
    """Global Attention Class"""

    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        """
        Args:
            encoder_hidden_dim:
            decoder_hidden_dim:
        """
        super().__init__()
        self.key_dim = encoder_hidden_dim * 2  # since bidirectional
        self.query_dim = decoder_hidden_dim

    def compute_score(self, query, encoder_outputs, mask):
        """
        Args:
            query: [batch_size, decoder_hidden_dim], previous decoder hidden
                state
            encoder_outputs: [src_length, batch_size, encoder_hidden_dim * 2],
                encoder outputs as keys and values
            mask: source input mask for masking out <pad> symbols
        """
        raise NotImplementedError

    @staticmethod
    def compute_context(scores, values):
        """Compute the weighed encoder_outputs according to the attention scores
        Args:
            scores: [batch_size, 1, src_length]
            values: [batch_size, src_lengths, encoder_hidden_dim * 2],
                encoder_outputs.permute(1, 0, 2)
        """
        weighted_encoder_outputs = torch.bmm(scores, values)  # [batch_size, 1, encoder_hid_dim * 2]
        weighted_encoder_outputs = weighted_encoder_outputs.permute(1, 0, 2)  # [1, batch_size, encoder_hid_dim * 2]
        return weighted_encoder_outputs

    def forward(self, query, encoder_outputs, mask):
        """
        Args:
            query: [batch_size, decoder_hidden_dim], previous decoder hidden
                state
            encoder_outputs: [src_length, batch_size, encoder_hidden_dim * 2],
                encoder outputs as keys and values
            mask: source input mask for masking out <pad> symbols
        """
        raise NotImplementedError
