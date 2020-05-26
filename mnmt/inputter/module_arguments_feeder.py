import torch.nn as nn


class ModuleArgsFeeder:

    def __init__(self, input_dim, embedding_dim,
                 hidden_dim, embedding_dropout, rnn_type: str,
                 num_layers, rnn_dropout):
        """
        Args:
            input_dim: vocabulary size
            embedding_dim: embedding layer output dimension
            hidden_dim: hidden dimension
            embedding_dropout: dropout values for embedding layer
            rnn_type (str): "LSTM" or "GRU"
            num_layers: number of layers in RNN
            rnn_dropout: dropout values for RNN
        """
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding_dropout = embedding_dropout
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.rnn_dropout = rnn_dropout

    def basic_embedding(self):
        return nn.Sequential(
            nn.Embedding(self.input_dim, self.embedding_dim),
            nn.Dropout(self.embedding_dropout)
        )

    def basic_rnn(self, in_dim, hidden_dim, bidirectional):
        """
        Args:
            in_dim: rnn input dimension
            hidden_dim: rnn hidden dimension
            bidirectional: indicate biRNN or not
        """
        return getattr(nn, self.rnn_type)(in_dim, hidden_dim, bidirectional=bidirectional,
                                          num_layers=self.num_layers, dropout=self.rnn_dropout)

