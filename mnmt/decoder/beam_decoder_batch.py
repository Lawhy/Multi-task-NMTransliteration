from mnmt.decoder import BasicDecoder
from mnmt.trainer.utils import inflate
import random
import torch


class BeamDecoderBatch(BasicDecoder):

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

    def training_forward(self, trg, encoder_outputs, encoder_final_state, mask, teacher_forcing_ratio):
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

    def forward(self, trg, encoder_outputs, encoder_final_state, mask, teacher_forcing_ratio):
        """
        Args:
            trg: [trg_length, batch_size], target samples batch
            encoder_outputs: [src_length, batch_size, encoder_hidden_dim * 2]
            encoder_final_state: [batch_size, encoder_hidden_dim * 2]
            mask: [batch_size, src_length], mask out <pad> for attention
            teacher_forcing_ratio: probability of applying teacher forcing or not
        """
        if not self.turn_on_beam:
            return self.training_forward(trg, encoder_outputs, encoder_final_state, mask, teacher_forcing_ratio)

        batch_size = trg.shape[1]
        y_hat = self.init_decoder_outputs(trg)  # [trg_length, batch_size, trg_vocab_size (input_dim)]
        s_t = self.init_s_0(encoder_final_state)
        input_t = trg[0, :]  # first input to the decoder is the <sos> tokens

        # inflate the matrices
        input_t = inflate(input_t, self.beam_size, dim=0)  # [batch * beam]
        assert input_t.shape[0] == self.beam_size * batch_size
        if isinstance(s_t, tuple):
            s_t = (inflate(s_t[0], self.beam_size, dim=0),
                   inflate(s_t[1], self.beam_size, dim=0))  # [batch * beam, hidden]
        else:
            s_t = inflate(s_t, self.beam_size, dim=0)
        y_hat = inflate(y_hat, self.beam_size, dim=1)
        inflated_encoder_outputs = inflate(encoder_outputs, self.beam_size, dim=1)  # [src, batch * beam, enc_hid * 2]
        inflated_mask = inflate(mask, self.beam_size, dim=0)
        scores_topk = torch.zeros(batch_size, self.beam_size).to(self.device)
        indices_topk = []

        # decode each sample in the batch
        # indexing: i for batch, t for time-step, j for node
        for t in range(1, trg.size(0)):
            # start from 1 as the first column are zeros that represent <sos>
            # each time using current y_t, attention, and previous s_{t-1}
            # to compute s_t and predict y_{t+1}_hat
            # we use the same subscript t for y and s here because y starts from 1, s starts from 0

            y_hat_t, s_t, _ = self.feed_forward_decoder(input_t, s_t, inflated_encoder_outputs, inflated_mask)
            # [batch * beam, trg_vocab_size OR hidden_dim]

            # Example to explain the following expansion
            # a = [[1, 2, 3]] => a.t() = [[1], [2], [3]] => a.expand(-1, v) = [[1]*v, [2]*v, [3]*v]
            # => reshape(1, v) = [[1..., 2..., 3...]] of size (1, 3 * v),
            # here beam = 3, trg-vocab-size = v, batch-size = 1

            scores_t = scores_topk.t().expand(-1, self.trg_vocab_size) \
                .reshape(batch_size, self.beam_size * self.trg_vocab_size) + \
                y_hat_t.reshape(batch_size, self.beam_size * self.trg_vocab_size)

            print(scores_t)
            scores_topk, indices_topk = torch.topk(scores_t, dim=1, k=self.beam_size)  # [batch_size, beam_size]
            y_hat_t_out = y_hat_t.clone()
            if isinstance(s_t, tuple):
                s_t_out = (s_t[0].clone(), s_t[1].clone())
            else:
                s_t_out = s_t.clone()

            for i in range(batch_size):

                # ind // trg_vocab_size gives beam number, times i directing to ith batch
                beam_inds = [ind // self.trg_vocab_size for ind in indices_topk[i]] * i
                beam_inds = torch.tensor(beam_inds).to(self.device)

                # next input takes topk indices, transformed to original vocab indices
                input_t[i*self.beam_size: (i+1)*self.beam_size] = (indices_topk[i, :] % self.trg_vocab_size)

                # reconstruct y_hat_t and s_t
                y_hat_t_out[i*self.beam_size: (i+1)*self.beam_size, :] = \
                    torch.index_select(y_hat_t, 0, beam_inds)

                if isinstance(s_t, tuple):
                    s_t_out[0][i*self.beam_size: (i+1)*self.beam_size] = torch.index_select(s_t[0], 0, beam_inds)
                    s_t_out[1][i*self.beam_size: (i+1)*self.beam_size] = torch.index_select(s_t[1], 0, beam_inds)
                else:
                    s_t_out[i*self.beam_size: (i+1)*self.beam_size] = torch.index_select(s_t, 0, beam_inds)

                # update hidden state sent to next time step
                s_t = s_t_out

            y_hat[t, :, :] = y_hat_t_out

        # backtrace

        return y_hat
