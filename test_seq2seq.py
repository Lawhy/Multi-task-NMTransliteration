from mnmt.encoder import BasicEncoder
from mnmt.decoder import BasicFeedForwardDecoder
from mnmt.decoder import GreedyDecoder
from mnmt.attention import AdditiveAttention
from mnmt.decoder import BridgeLayer
from mnmt.model import Seq2Seq
from mnmt.datasets import *
from mnmt.inputter import ArgsFeeder
from mnmt.inputter import ModuleArgsFeeder
from mnmt.inputter import TrainerArgsFeeder
from mnmt.trainer.utils import *


def set_up_args(dataset):
    build_vocabs(dataset, dict_min_freqs={'en': 1, 'ch': 1, 'pinyin_str': 1, 'pinyin_char': 1})
    input_dim, output_dim = 0, 0
    for name, field in dataset.fields:
        if name == 'en':
            input_dim = len(field.vocab)
            src_pad_idx = field.vocab.stoi[field.pad_token]
        elif name == 'ch':
            output_dim = len(field.vocab)

    enc_args_feeder = ModuleArgsFeeder(input_dim=input_dim, embedding_dim=256, hidden_dim=512,
                                       embedding_dropout=0.1, rnn_type='LSTM',
                                       num_layers=2, rnn_dropout=0.2)
    dec_args_feeder = ModuleArgsFeeder(input_dim=output_dim, embedding_dim=256, hidden_dim=512,
                                       embedding_dropout=0.1, rnn_type='LSTM',
                                       num_layers=2, rnn_dropout=0.2)
    trainer_args_feeder = TrainerArgsFeeder()
    args_feeder = ArgsFeeder(enc_args_feeder, dec_args_feeder, trainer_args_feeder,
                             batch_size=64, src_pad_idx=src_pad_idx)
    return args_feeder


def test_seq2seq(args_feeder):
    encoder = BasicEncoder(args_feeder)
    feed_forward_decoder = \
        BasicFeedForwardDecoder(args_feeder,
                                AdditiveAttention(encoder_hidden_dim=args_feeder.encoder_args_feeder.hidden_dim,
                                                  decoder_hidden_dim=args_feeder.decoder_args_feeder.hidden_dim))
    bridge_layer = BridgeLayer(encoder_hidden_dim=args_feeder.encoder_args_feeder.hidden_dim,
                               decoder_hidden_dim=args_feeder.decoder_args_feeder.hidden_dim,
                               num_of_states=2)
    decoder = GreedyDecoder(feed_forward_decoder, bridge_layer, device=args_feeder.device)
    model = Seq2Seq(args_feeder, encoder, decoder, teacher_forcing_ratio=0.8)
    print(model.apply(init_weights))
    count_parameters(model)
    return model


if __name__ == '__main__':
    dataset = DICT['dataset']
    args_feeder = set_up_args(dataset)
    model = test_seq2seq(args_feeder)
