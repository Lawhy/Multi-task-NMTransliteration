import torch
import torch.nn as nn
import torch.nn.functional as F
from mnmt.attention import Attention


class AdditiveAttention(Attention):
    """Bahdanau Attention (mlp)"""

    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        """
        Args:
            encoder_hidden_dim:
            decoder_hidden_dim:
        """
        super().__init__(encoder_hidden_dim, decoder_hidden_dim)
        self.additive_mapping = nn.Linear((encoder_hidden_dim * 2) + decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def compute_score(self, query, encoder_outputs, mask):
        """
        Args:
            query: [batch_size, decoder_hidden_dim], previous decoder hidden
                state
            encoder_outputs: [src_length, batch_size, encoder_hidden_dim * 2],
                encoder outputs as keys and values
            mask: source input mask for masking out <pad> symbols
        """
        src_length = encoder_outputs.shape[0]  # maximum source length of a batch

        # repeat query (decoder hidden state) src_length times
        query = query.unsqueeze(1).repeat(1, src_length, 1)  # [batch_size, src_length, decoder_hidden_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch_size, src_lengths, encoder_hidden_dim * 2]

        # the additive mapping essential does Wq + Uh, W ks query mapping, U is key mapping
        energy = torch.tanh(self.additive_mapping(
            torch.cat((query, encoder_outputs), dim=2)))  # [batch_size, src_length, decoder_hidden_dim]

        raw_attn_scores = self.v(energy).squeeze(2)  # [batch_size, src_length]
        # e^{-Inf} = 0, so ignore <pad> in softmax
        raw_attn_scores = raw_attn_scores.masked_fill(mask == 0, -float('inf'))
        attn_scores = F.softmax(raw_attn_scores, dim=1).unsqueeze(1)  # scores = [batch_size, 1, src_length]

        return attn_scores

    def forward(self, query, encoder_outputs, mask):
        """
        Args:
            query: [batch_size, decoder_hidden_dim], previous decoder hidden
                state
            encoder_outputs: [src_length, batch_size, encoder_hidden_dim * 2],
                encoder outputs as keys and values
            mask: source input mask for masking out <pad> symbols
        """
        assert query.size(0) == encoder_outputs.size(1)
        scores = self.compute_score(query, encoder_outputs, mask)  # scores = [batch_size, 1, src_length]
        values = encoder_outputs.permute(1, 0, 2)  # [batch_size, src_lengths, encoder_hidden_dim * 2]
        # context matrix is the weighted encoder outputs for each sample in the batch
        print(scores.size(), values.size())
        context = super().compute_context(scores, values)  # [1, batch_size, encoder_hid_dim * 2]

        return scores.squeeze(1), context
