from mnmt.inputter import ArgsFeeder
import torch
import torch.nn as nn


class BasicFeedForwardDecoder(nn.Module):
    """Single-step Decoder"""
    def __init__(self, args_feeder: ArgsFeeder, attention, decoder_index=0):
        """
        Args:
            args_feeder (ArgsFeeder): the general arguments feeder for the
                entire model
            attention: Attention instance
            decoder_index: indicate which decoder argument to use
        """
        super().__init__()

        self.attrs = args_feeder.decoder_args_feeders[decoder_index]  # decoder's attributes
        self.batch_size = args_feeder.batch_size
        self.trg_eos_idx = args_feeder.trg_eos_idx
        self.attention = attention
        self.embedding = self.attrs.basic_embedding()
        rnn_input_dim = self.attrs.embedding_dim + (args_feeder.encoder_args_feeder.hidden_dim * 2)
        self.rnn = self.attrs.basic_rnn(in_dim=rnn_input_dim,
                                        hidden_dim=self.attrs.hidden_dim,
                                        bidirectional=False)
        # [y_t, attn_t, s_t] -> y_{t+1}
        self.prediction = nn.Sequential(
            nn.Linear(self.attrs.embedding_dim + self.attention.key_dim + self.attrs.hidden_dim, self.attrs.input_dim),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, y_t, s_t_minus_1, encoder_outputs, mask, ith_sample=None):
        """s_t = Decoder(d(y_t), w_t, s_{t-1}) decoder hidden state for time t
            y_t_plus_1_hat = f(d(y_t), w_t, s_t) decoder output for time t

        Args:
            y_t: [batch_size] target one-hot embedding for time step t
            s_t_minus_1: [batch_size, hidden_dim] or tuple if LSTM
            encoder_outputs: [src_length, batch_size, encoder_hidden_dim * 2]
            mask: [batch_size, src_length]
            ith_sample:
        """
        y_t = y_t.unsqueeze(0)  # y_t = [1, batch_size]
        y_t = self.embedding(y_t)  # y_t = [1, batch_size, embedding_dim], dropout applied

        # context = [1, batch_size, encoder_hidden_dim * 2]
        # rnn_output = [1, batch_size, hidden_dim]
        # s_t = [n_layers, batch_size, hidden_dim], tuple for LSTM

        # for beam search setting
        if ith_sample is not None:
            mask = mask[ith_sample, :].unsqueeze(0)  # match batch-size 1

        if isinstance(s_t_minus_1, tuple):  # LSTM
            scores, context = self.attention(s_t_minus_1[0], encoder_outputs, mask)
            s_0, c_0 = s_t_minus_1[0].unsqueeze(0).repeat(2, 1, 1), s_t_minus_1[1].unsqueeze(0).repeat(2, 1, 1)
            rnn_output, (s_t, c_t) = self.rnn(torch.cat((y_t, context), dim=2), (s_0, c_0))
            c_t = c_t[-1]  # [batch_size, hidden_dim]
        else:  # GRU
            scores, context = self.attention(s_t_minus_1, encoder_outputs, mask)
            s_0 = s_t_minus_1.unsqueeze(0).repeat(2, 1, 1)
            rnn_output, s_t = self.rnn(torch.cat((y_t, context), dim=2), s_0)
        s_t = s_t[-1]  # [batch size, hidden_dim], as last layer extracted
        assert (rnn_output == s_t.unsqueeze(0)).all()
        y_t = y_t.squeeze(0)  # [batch_size]
        context = context.squeeze(0)  # [batch_size, encoder_hidden_dim * 2]

        # make prediction [y_t, attn_t, s_t] -> y_{t+1}
        y_t_plus_1_hat = self.prediction(
            torch.cat((y_t, context, s_t), dim=1))  # [batch_size, target_input_dim]

        # concat decoder hidden and cell state for LSTM
        if isinstance(s_t_minus_1, tuple):
            s_t = (s_t, c_t)

        return y_t_plus_1_hat, s_t, scores
