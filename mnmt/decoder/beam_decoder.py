from mnmt.decoder import BasicDecoder
from mnmt.trainer.utils import inflate
import random
import torch


class BeamDecoder(BasicDecoder):

    def __init__(self, feed_forward_decoder, bridge_layer, device, beam_size):
        """
        Args:
            feed_forward_decoder:
            bridge_layer:
            device:
        """
        super().__init__(feed_forward_decoder, bridge_layer, device)
        self.beam_size = beam_size
        self.hidden_dim = self.feed_forward_decoder.attrs.hidden_dim
        self.batch_size = self.feed_forward_decoder.attrs.batch_size
        self.pos_index = (torch.tensor(range(self.batch_size)) * self.beam_size).to(self.device).view(-1, 1)
        self.EOS = self.feed_forward_decoder.attrs.trg_eos_idx

    def forward(self, trg, encoder_outputs, encoder_final_state, mask, teacher_forcing_ratio):
        """
        Args:
            trg: [trg_length, batch_size], target samples batch
            encoder_outputs: [src_length, batch_size, encoder_hidden_dim * 2]
            encoder_final_state: [batch_size, encoder_hidden_dim * 2]
            mask: [batch_size, src_length], mask out <pad> for attention
            teacher_forcing_ratio: probability of applying teacher forcing or not
        """
        batch_size = encoder_outputs.shape[1]
        s_t = self.init_s_0(encoder_final_state)  # [batch, hidden] or tuple
        if isinstance(s_t, tuple):
            s_t = (inflate(s_t[0], times=self.beam_size, dim=0),
                   inflate(s_t[1], times=self.beam_size, dim=0))
        else:
            s_t = inflate(s_t, self.beam_size, dim=0)
        # s_t: [batch * beam, hidden] or tuple
        y_hat_t = inflate(trg[0, :], times=self.beam_size, dim=0)  # [batch * beam]

        # Initialize the scores; for the first step,
        # ignore the inflated copies to avoid duplicate entries in the top k
        sequence_scores = torch.tensor((batch_size * self.beam_size, 1)).to(self.device)
        sequence_scores.fill_(-float('Inf'))
        sequence_scores.index_fill_(0, torch.tensor([i * self.beam_size for i in range(0, batch_size)]), 0.0)

        # Store decisions for backtracking
        stored_outputs = list()
        stored_scores = list()
        stored_predecessors = list()
        stored_emitted_symbols = list()
        stored_hidden = list()

        # inflate encoder outputs and mask
        inflated_encoder_outputs = inflate(encoder_outputs, self.beam_size, dim=1)
        inflated_mask = inflate(mask, self.beam_size, dim=0)

        for t in range(1, trg.size(0)):

            # Run the RNN one step forward
            log_softmax_output, s_t, _ = self.feed_forward_decoder(y_hat_t, s_t,
                                                                   inflated_encoder_outputs, inflated_mask)

            # If doing local backprop (e.g. supervised training), retain the output layer
            stored_outputs.append(log_softmax_output)

            # To get the full sequence scores for the new candidates,
            # add the local scores for t_i to the predecessor scores for t_(i-1)
            sequence_scores = inflate(sequence_scores, self.trg_vocab_size, 1)
            sequence_scores += log_softmax_output.squeeze(1)
            scores, candidates = sequence_scores.view(batch_size, -1).topk(self.beam_size, dim=1)

            # Reshape input = (bk, 1) and sequence_scores = (bk, 1)
            y_hat_t = (candidates % self.trg_vocab_size).view(batch_size * self.beam_size, 1)
            # Apply teacher forcing
            if random.random() < teacher_forcing_ratio:
                y_hat_t = inflate(trg[t], self.beam_size, dim=0)
            sequence_scores = scores.view(batch_size * self.beam_size, 1)

            # Update fields for next time step
            predecessors = (candidates / self.trg_vocab_size + self.pos_index.expand_as(candidates))\
                .view(batch_size * self.beam_size, 1)
            if isinstance(s_t, tuple):
                s_t = tuple([h.index_select(1, predecessors.squeeze()) for h in s_t])
            else:
                s_t = s_t.index_select(1, predecessors.squeeze())

            # Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
            stored_scores.append(sequence_scores.clone())
            eos_indices = y_hat_t.data.eq(self.EOS)
            if eos_indices.nonzero().dim() > 0:
                sequence_scores.data.masked_fill_(eos_indices, -float('inf'))

            # Cache results for backtracking
            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(y_hat_t)
            stored_hidden.append(s_t)

        # Do backtracking to return the optimal values
        output, h_t, h_n, s, l, p = self._backtrack(stored_outputs, stored_hidden,
                                                    stored_predecessors, stored_emitted_symbols,
                                                    stored_scores, batch_size, self.hidden_dim)

        # Build return objects
        decoder_outputs = [step[:, 0, :] for step in output]
        # if isinstance(h_n, tuple):
        #     decoder_hidden = tuple([h[:, :, 0, :] for h in h_n])
        # else:
        #     decoder_hidden = h_n[:, :, 0, :]

        return decoder_outputs

    def _backtrack(self, nw_output, nw_hidden, predecessors, symbols, scores, b, hidden_size):
        """Backtracks over batch to generate optimal k-sequences.
        Args:
            nw_output [(batch*k, vocab_size)] * sequence_length: A Tensor of outputs from network
            nw_hidden [(num_layers, batch*k, hidden_size)] * sequence_length: A Tensor of hidden states from network
            predecessors [(batch*k)] * sequence_length: A Tensor of predecessors
            symbols [(batch*k)] * sequence_length: A Tensor of predicted tokens
            scores [(batch*k)] * sequence_length: A Tensor containing sequence scores for every token t = [0, ... , seq_len - 1]
            b: Size of the batch
            hidden_size: Size of the hidden state
        Returns:
            output [(batch, k, vocab_size)] * sequence_length: A list of the output probabilities (p_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
            h_t [(batch, k, hidden_size)] * sequence_length: A list containing the output features (h_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
            h_n(batch, k, hidden_size): A Tensor containing the last hidden state for all top-k sequences.
            score [batch, k]: A list containing the final scores for all top-k sequences
            length [batch, k]: A list specifying the length of each sequence in the top-k candidates
            p (batch, k, sequence_len): A Tensor containing predicted sequence
        """

        lstm = isinstance(nw_hidden[0], tuple)

        # initialize return variables given different types
        output = list()
        h_t = list()
        p = list()
        # Placeholder for last hidden state of top-k sequences.
        # If a (top-k) sequence ends early in decoding, `h_n` contains
        # its hidden state when it sees EOS.  Otherwise, `h_n` contains
        # the last hidden state of decoding.
        if lstm:
            state_size = nw_hidden[0][0].size()
            h_n = tuple([torch.zeros(state_size), torch.zeros(state_size)])
        else:
            h_n = torch.zeros(nw_hidden[0].size())
        l = [[self.rnn.max_length] * self.beam_size for _ in range(b)]  # Placeholder for lengths of top-k sequences
        # Similar to `h_n`

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = scores[-1].view(b, self.beam_size).topk(self.beam_size)
        # initialize the sequence scores with the sorted last step beam scores
        s = sorted_score.clone()

        batch_eos_found = [0] * b  # the number of EOS found
        # in the backward loop below for each batch

        t = self.rnn.max_length - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add self.pos_index for indexing variable with b*k as the first dimension.
        t_predecessors = (sorted_idx + self.pos_index.expand_as(sorted_idx)).view(b * self.beam_size)
        while t >= 0:
            # Re-order the variables with the back pointer
            current_output = nw_output[t].index_select(0, t_predecessors)
            if lstm:
                current_hidden = tuple([h.index_select(1, t_predecessors) for h in nw_hidden[t]])
            else:
                current_hidden = nw_hidden[t].index_select(1, t_predecessors)
            current_symbol = symbols[t].index_select(0, t_predecessors)
            # Re-order the back pointer of the previous step with the back pointer of
            # the current step
            t_predecessors = predecessors[t].index_select(0, t_predecessors).squeeze()

            # This tricky block handles dropped sequences that see EOS earlier.
            # The basic idea is summarized below:
            #
            #   Terms:
            #       Ended sequences = sequences that see EOS early and dropped
            #       Survived sequences = sequences in the last step of the beams
            #
            #       Although the ended sequences are dropped during decoding,
            #   their generated symbols and complete backtracking information are still
            #   in the backtracking variables.
            #   For each batch, everytime we see an EOS in the backtracking process,
            #       1. If there is survived sequences in the return variables, replace
            #       the one with the lowest survived sequence score with the new ended
            #       sequences
            #       2. Otherwise, replace the ended sequence with the lowest sequence
            #       score with the new ended sequence
            #
            eos_indices = symbols[t].data.squeeze(1).eq(self.EOS).nonzero()
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0) - 1, -1, -1):
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = int(idx[0] / self.beam_size)
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = self.beam_size - (batch_eos_found[b_idx] % self.beam_size) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * self.beam_size + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = predecessors[t][idx[0]]
                    current_output[res_idx, :] = nw_output[t][idx[0], :]
                    if lstm:
                        current_hidden[0][:, res_idx, :] = nw_hidden[t][0][:, idx[0], :]
                        current_hidden[1][:, res_idx, :] = nw_hidden[t][1][:, idx[0], :]
                        h_n[0][:, res_idx, :] = nw_hidden[t][0][:, idx[0], :].data
                        h_n[1][:, res_idx, :] = nw_hidden[t][1][:, idx[0], :].data
                    else:
                        current_hidden[:, res_idx, :] = nw_hidden[t][:, idx[0], :]
                        h_n[:, res_idx, :] = nw_hidden[t][:, idx[0], :].data
                    current_symbol[res_idx, :] = symbols[t][idx[0]]
                    s[b_idx, res_k_idx] = scores[t][idx[0]].data[0]
                    l[b_idx][res_k_idx] = t + 1

            # record the back tracked results
            output.append(current_output)
            h_t.append(current_hidden)
            p.append(current_symbol)

            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        s, re_sorted_idx = s.topk(self.beam_size)
        for b_idx in range(b):
            l[b_idx] = [l[b_idx][k_idx.item()] for k_idx in re_sorted_idx[b_idx, :]]

        re_sorted_idx = (re_sorted_idx + self.pos_index.expand_as(re_sorted_idx)).view(b * self.beam_size)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        output = [step.index_select(0, re_sorted_idx).view(b, self.beam_size, -1) for step in reversed(output)]
        p = [step.index_select(0, re_sorted_idx).view(b, self.beam_sizek, -1) for step in reversed(p)]
        if lstm:
            h_t = [tuple([h.index_select(1, re_sorted_idx).view(-1, b, self.beam_size, hidden_size) for h in step]) for step in
                   reversed(h_t)]
            h_n = tuple([h.index_select(1, re_sorted_idx.data).view(-1, b, self.beam_size, hidden_size) for h in h_n])
        else:
            h_t = [step.index_select(1, re_sorted_idx).view(-1, b, self.beam_size, hidden_size) for step in reversed(h_t)]
            h_n = h_n.index_select(1, re_sorted_idx.data).view(-1, b, self.beam_size, hidden_size)
        s = s.data

        return output, h_t, h_n, s, l, p

    @staticmethod
    def _mask_symbol_scores(score, idx, masking_score=-float('inf')):
        score[idx] = masking_score

    @staticmethod
    def _mask(tensor, idx, dim=0, masking_score=-float('inf')):
        if len(idx.size()) > 0:
            indices = idx[:, 0]
            tensor.index_fill_(dim, indices, masking_score)
