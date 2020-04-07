from mnmt.encoder import BasicEncoder
from mnmt.decoder import BasicFeedForwardDecoder
from mnmt.decoder import GreedyDecoder
from mnmt.attention import AdditiveAttention
from mnmt.decoder import BridgeLayer
from mnmt.model import Seq2Seq
from mnmt.datasets import *
from mnmt.inputter import ArgsFeeder
from mnmt.inputter import ModuleArgsFeeder
from mnmt.trainer.utils import *
from mnmt.trainer import Trainer


def set_up_args(data_container):
    build_vocabs(data_container, dict_min_freqs={'en': 1, 'ch': 1, 'pinyin_str': 1, 'pinyin_char': 1})
    for name, field in data_container.fields:
        if name == 'en':
            input_dim = len(field.vocab)
            src_pad_idx = field.vocab.stoi[field.pad_token]
        elif name == 'ch':
            output_dim = len(field.vocab)
            trg_pad_idx = field.vocab[field.pad_token]

    enc_args_feeder = ModuleArgsFeeder(input_dim=input_dim, embedding_dim=256, hidden_dim=512,
                                       embedding_dropout=0.1, rnn_type='LSTM',
                                       num_layers=2, rnn_dropout=0.2)
    dec_args_feeder = ModuleArgsFeeder(input_dim=output_dim, embedding_dim=256, hidden_dim=512,
                                       embedding_dropout=0.1, rnn_type='LSTM',
                                       num_layers=2, rnn_dropout=0.2)
    return ArgsFeeder(enc_args_feeder, dec_args_feeder,
                      batch_size=64, src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx,
                      optim_choice='Adam', learning_rate=0.003, decay_patience=0,
                      lr_decay_factor=0.9, valid_criterion='ACC', early_stopping_patience=1000,
                      total_epochs=100, report_interval=50, exp_num=1, multi_task_ratio=1, data_container=data_container,
                      src_lang='en', trg_lang='ch', auxiliary_name='pinyin_str', quiet_translate=True)


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
    model = Seq2Seq(args_feeder, encoder, decoder, teacher_forcing_ratio=0.8).to(args_feeder.device)
    print(model.apply(init_weights))
    count_parameters(model)
    return model


if __name__ == '__main__':
    set_reproducibility(seed=1234)
    dict_dataset = DICT['data_container']
    seq2seq_args_feeder = set_up_args(dict_dataset)
    test_model = test_seq2seq(seq2seq_args_feeder)
    test_trainer = Trainer(seq2seq_args_feeder, test_model)
    test_trainer.run(burning_epoch=0)
