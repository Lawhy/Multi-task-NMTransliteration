from mnmt.decoder import BasicDecoder
import torch
import random


class BeamDecoder(BasicDecoder):

    def __init__(self, feed_forward_decoder, bridge_layer, device, beam_size, turn_on_beam=False):
        """
        Args:
            feed_forward_decoder:
            bridge_layer:
            device:
        """
        super().__init__(feed_forward_decoder, bridge_layer, device)
        self.beam_size = beam_size
        self.hidden_dim = self.feed_forward_decoder.attrs.hidden_dim
        self.turn_on_beam = turn_on_beam
        self.eos_idx = self.feed_forward_decoder.trg_eos_idx

    def forward(self, trg, encoder_outputs, encoder_final_state, mask, teacher_forcing_ratio):
        """
        Args:
            trg: [trg_length, batch_size], target samples batch
            encoder_outputs: [src_length, batch_size, encoder_hidden_dim * 2]
            encoder_final_state: [batch_size, encoder_hidden_dim * 2]
            mask: [batch_size, src_length], mask out <pad> for attention
            teacher_forcing_ratio: probability of applying teacher forcing or not
        """
        y_hat = self.init_decoder_outputs(trg)  # [trg_length, batch_size, trg_vocab_size (input_dim)]
        s_t = self.init_s_0(encoder_final_state)
        y_hat_t = trg[0, :]  # first input to the decoder is the <sos> tokens

        for t in range(1, trg.size(0)):
            # start from 1 as the first column are zeros that represent <sos>
            # each time using current y_t, attention, and previous s_{t-1}
            # to compute s_t and predict y_{t+1}_hat
            # we use the same subscript t for y and s here because y starts from 1, s starts from 0
            y_hat_t, s_t, _ = self.feed_forward_decoder(y_hat_t, s_t, encoder_outputs, mask)
            y_hat[t] = y_hat_t
            # greedy strategy as only top1 prediction considered
            teacher_force = random.random() < teacher_forcing_ratio
            y_hat_t = trg[t] if teacher_force else y_hat_t.argmax(1)
            assert y_hat_t.size() == trg[t].size()

        return y_hat

    @staticmethod
    def greedy_decode(decoder_hidden, encoder_outputs, target_tensor):
        """
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        """

        batch_size, seq_len = target_tensor.size()
        decoded_batch = torch.zeros((batch_size, MAX_LENGTH))
        decoder_input = torch.LongTensor([[SOS_token] for _ in range(batch_size)], device=device)

        for t in range(MAX_LENGTH):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.data.topk(1)  # get candidates
            topi = topi.view(-1)
            decoded_batch[:, t] = topi

            decoder_input = topi.detach().view(-1, 1)

        return decoded_batch
