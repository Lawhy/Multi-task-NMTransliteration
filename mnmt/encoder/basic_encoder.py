import torch
import torch.nn as nn
from mnmt.inputter import ArgsFeeder


class BasicEncoder(nn.Module):
    def __init__(self, args_feeder: ArgsFeeder):
        """
        Args:
            args_feeder (ArgsFeeder):
        """
        super().__init__()
        self.encoder_args_feeder = args_feeder.encoder_args_feeder
        self.embedding = self.encoder_args_feeder.basic_embedding()
        self.rnn = self.encoder_args_feeder.basic_rnn(in_dim=self.encoder_args_feeder.embedding_dim,
                                                      hidden_dim=self.encoder_args_feeder.hidden_dim,
                                                      bidirectional=True)  # num_directions = 2

    def forward(self, src, src_lens):
        """
        Args:
            src: [src_length, batch_size], source inputs in a batch
            src_lens: [batch_size], source input length for each sample in the batch
        """
        assert src.size(1) == src_lens.size()  # should all be batch_size

        src_embedding = self.embedding(src)  # [src_length, batch_size, embedding_dim]
        src_embedding_packed = nn.utils.rnn.pack_padded_sequence(src_embedding, src_lens)

        # rnn_outputs_packed is a packed sequence containing all last layer hidden states
        # h_n is now from the final non-padded elements (for all layers) in the batch

        # h_n = [n_layers * 2, batch_size, hidden_dim]
        rnn_outputs_packed, h_n = self.rnn(src_embedding_packed)
        # encoder_outputs = [src_length, batch_size, hidden_dim * 2]
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs_packed)
        # final_state = [batch_size, hidden_dim * 2]
        final_state = self.extract_final_state(h_n, batch_size=src.size(1),
                                               hidden_dim=self.encoder_args_feeder.hidden_dim)

        # encoder_outputs = [src_length, batch_size, hidden_dim * 2]
        return encoder_outputs, final_state

    def extract_final_state(self, h_n, batch_size, hidden_dim):
        """
        Args:
            h_n: standard output from nn.LSTM or nn.GRU
                if GRU:
                    [-2, :, : ] is the last of the forwards RNN [-1, :, : ] is
                    the last of the backwards RNN [n_layers * 2, batch_size,
                    hidden_dim] -> [batch_size, hidden_dim * 2]
                else:
                    have done extra step for the tuple (LSTM)
            batch_size: number of samples in one batch
            hidden_dim: encoder hidden dimension
        """
        if isinstance(h_n, tuple):  # For LSTM: (h_n, c_n)
            assert self.encoder_args_feeder.rnn_type == "LSTM"
            final_state = (torch.cat((h_n[0][-2, :, :], h_n[0][-1, :, :]), dim=1),
                           torch.cat((h_n[1][-2, :, :], h_n[1][-1, :, :]), dim=1))
            assert final_state[0].size() == (batch_size, hidden_dim * 2)  # check hidden state
            assert final_state[1].size(1) == (batch_size, hidden_dim * 2)  # check cell state
        else:
            final_state = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
            assert final_state.size() == (batch_size, hidden_dim * 2)  # check hidden state
        return final_state
