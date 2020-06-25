from mnmt.encoder import BasicEncoder
from mnmt.decoder import BasicFeedForwardDecoder
from mnmt.decoder import BeamDecoder
from mnmt.attention import AdditiveAttention
from mnmt.decoder import BridgeLayer
from mnmt.model import Seq2MultiSeq
from mnmt.datasets import *
from mnmt.inputter import ArgsFeeder
from mnmt.inputter import ModuleArgsFeeder
from mnmt.trainer.utils import *
from mnmt.trainer import Trainer
import torch.nn as nn
import os


def set_up_args(data_container, exp_num):
    build_vocabs(data_container, dict_min_freqs={'en': 1, 'ch': 1, 'pinyin_str': 1, 'pinyin_char': 1})
    for name, field in data_container.fields:
        if name == 'en':
            input_dim = len(field.vocab)
            src_pad_idx = field.vocab.stoi[field.pad_token]
        elif name == 'ch':
            output_dim = len(field.vocab)
            trg_pad_idx = field.vocab[field.pad_token]
            trg_eos_idx = field.vocab[field.eos_token]

    enc_args_feeder = ModuleArgsFeeder(input_dim=input_dim, embedding_dim=256, hidden_dim=512,
                                       embedding_dropout=0.1, rnn_type='LSTM',
                                       num_layers=2, rnn_dropout=0.2)
    dec_args_feeder = ModuleArgsFeeder(input_dim=output_dim, embedding_dim=256, hidden_dim=512,
                                       embedding_dropout=0.1, rnn_type='LSTM',
                                       num_layers=2, rnn_dropout=0.2)
    dec_args_feeder_aux = ModuleArgsFeeder(input_dim=output_dim, embedding_dim=128, hidden_dim=256,
                                           embedding_dropout=0.1, rnn_type='LSTM',
                                           num_layers=2, rnn_dropout=0.1)
    return ArgsFeeder(enc_args_feeder, [dec_args_feeder, dec_args_feeder_aux],
                      batch_size=64, src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx,
                      optim_choice='Adam', learning_rate=0.003, decay_patience=0,
                      lr_decay_factor=0.9, valid_criterion='ACC', early_stopping_patience=100,
                      total_epochs=100, report_interval=50, exp_num=exp_num, multi_task_ratio=66,
                      data_container=data_container,
                      src_lang='en', trg_lang='ch', auxiliary_name='pinyin_str', quiet_translate=True,
                      valid_out_path=f"experiments/exp{exp_num}/valid.out",
                      test_out_path=f"experiments/exp{exp_num}/test.out",
                      beam_size=1, trg_eos_idx=trg_eos_idx)


def test_seq2seq(args_feeder):
    decoder_args_feeder = args_feeder.decoder_args_feeders[0]
    decoder_args_feeder_aux = args_feeder.decoder_args_feeders[1]

    encoder = BasicEncoder(args_feeder)

    feed_forward_decoder = \
        BasicFeedForwardDecoder(args_feeder,
                                AdditiveAttention(encoder_hidden_dim=args_feeder.encoder_args_feeder.hidden_dim,
                                                  decoder_hidden_dim=decoder_args_feeder.hidden_dim),
                                decoder_index=0)
    feed_forward_decoder_aux = \
        BasicFeedForwardDecoder(args_feeder,
                                AdditiveAttention(encoder_hidden_dim=args_feeder.encoder_args_feeder.hidden_dim,
                                                  decoder_hidden_dim=decoder_args_feeder_aux.hidden_dim),
                                decoder_index=1)

    bridge_layer = BridgeLayer(encoder_hidden_dim=args_feeder.encoder_args_feeder.hidden_dim,
                               decoder_hidden_dim=decoder_args_feeder.hidden_dim,
                               num_of_states=2)
    bridge_layer_aux = BridgeLayer(encoder_hidden_dim=args_feeder.encoder_args_feeder.hidden_dim,
                                   decoder_hidden_dim=decoder_args_feeder_aux.hidden_dim,
                                   num_of_states=2)

    decoder = BeamDecoder(feed_forward_decoder, bridge_layer, device=args_feeder.device,
                          beam_size=args_feeder.beam_size)
    decoder_aux = BeamDecoder(feed_forward_decoder_aux, bridge_layer_aux, device=args_feeder.device,
                              beam_size=args_feeder.beam_size)

    model = Seq2MultiSeq(args_feeder, encoder,
                         nn.ModuleList([decoder, decoder_aux]),
                         teacher_forcing_ratio=0.8).to(args_feeder.device)
    return model


if __name__ == '__main__':
    set_reproducibility(seed=1234)
    mtrs = [8 / 9, 5 / 6, 2 / 3, 1 / 2, 1 / 4, 1 / 6]
    aux_lang = "pinyin_str"
    mtrs.reverse()
    try:
        # NEWS Multi
        news_dataset = NEWS['data_container']
        for i in range(len(mtrs)):
            i += 1
            os.makedirs("experiments/exp" + str(60 + i), exist_ok=True)
            seq2seq_args_feeder = set_up_args(news_dataset, exp_num=60 + i)
            seq2seq_args_feeder.multi_task_ratio = mtrs[i - 1]
            seq2seq_args_feeder.auxiliary_name = aux_lang
            test_model = test_seq2seq(seq2seq_args_feeder)
            test_trainer = Trainer(seq2seq_args_feeder, test_model)
            test_trainer.run(burn_in_epoch=15)
            test_trainer.best_model_output(test_ref_dict=NEWS['test-set-dict'],
                                           beam_size=1, score_choice="O+N", length_norm_ratio=0.7)
            test_trainer.best_model_output(test_ref_dict=NEWS['test-set-dict'],
                                           beam_size=10, score_choice="O+N", length_norm_ratio=0.7)
    except KeyboardInterrupt:
        print("Exiting loop")
