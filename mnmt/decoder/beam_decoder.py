from mnmt.decoder import BasicDecoder
import random
import torch


class BeamNode:
    def __init__(self, y_hat_n, log_prob_path, s_n, pre_node):
        self.y_hat_n = y_hat_n
        self.log_prob_path = log_prob_path
        self.s_n = s_n
        self.pre_node = pre_node


class BeamDecoder(BasicDecoder):

    def __init__(self, feed_forward_decoder, bridge_layer, device, beam_size,
                 turn_on_beam=False, score_choice="bias", length_norm_ratio=0.7,
                 max_length=None):
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
        self.score_choice = score_choice
        self.length_norm_ratio = length_norm_ratio
        self.MAX_LENGTH = max_length

    def forward(self, trg, encoder_outputs, encoder_final_state, mask, teacher_forcing_ratio):
        """
        Args:
            trg: [trg_length, batch_size], target samples batch
            encoder_outputs: [src_length, batch_size, encoder_hidden_dim * 2]
            encoder_final_state: [batch_size, encoder_hidden_dim * 2]
            mask: [batch_size, src_length], mask out <pad> for attention
            teacher_forcing_ratio: probability of applying teacher forcing or not
        """
        MAX_LENGTH = self.MAX_LENGTH if self.MAX_LENGTH is not None else trg.size(0)
        batch_size = trg.shape[1]
        y_hat = self.init_decoder_outputs(MAX_LENGTH, batch_size)  # [trg_length, batch_size, trg_vocab_size]
        s_t = self.init_s_0(encoder_final_state)
        y_hat_t = trg[0, :]  # first input to the decoder is the <sos> tokens
        decoded_batch = torch.zeros([batch_size, MAX_LENGTH], dtype=torch.int32).to(self.device)

        for t in range(1, MAX_LENGTH):
            # start from 1 as the first column are zeros that represent <sos>
            # each time using current y_t, attention, and previous s_{t-1}
            # to compute s_t and predict y_{t+1}_hat
            # we use the same subscript t for y and s here because y starts from 1, s starts from 0
            y_hat_t, s_t, _ = self.feed_forward_decoder(y_hat_t, s_t, encoder_outputs, mask)
            y_hat[t] = y_hat_t  # log-prob
            # greedy strategy as only top1 prediction considered
            teacher_force = random.random() < teacher_forcing_ratio
            y_hat_t = trg[t] if teacher_force else y_hat_t.argmax(1)  # log-prob => index
            decoded_batch[:, t] = y_hat_t
            assert y_hat_t.size() == trg[t].size()

        if self.turn_on_beam and self.beam_size > 1:
            decoded_batch = self.beam_decode(trg, encoder_outputs, encoder_final_state, mask)

        return y_hat, decoded_batch

    def beam_decode(self, trg, encoder_outputs, encoder_final_state, mask):

        MAX_LENGTH = self.MAX_LENGTH if self.MAX_LENGTH is not None else trg.size(0)
        batch_size = trg.shape[1]
        s_t = self.init_s_0(encoder_final_state)
        y_hat_t = trg[0, :]  # first input to the decoder is the <sos> tokens
        decoded_batch = torch.zeros([batch_size, MAX_LENGTH], dtype=torch.int32).to(self.device)

        # decode each sample in the batch
        # indexing: i for batch, t for time-step, j for node
        for i in range(batch_size):
            y_hat_i_t = y_hat_t[i].unsqueeze(0)  # torch.Size([1]), all <sos> indeed

            if isinstance(s_t, tuple):
                s_i_t = (s_t[0][i].unsqueeze(0), s_t[1][i].unsqueeze(0))  # [1, hidden_dim], tuple
            else:
                s_i_t = s_t[i].unsqueeze(0)  # [1, hidden_dim]

            encoder_outputs_i = encoder_outputs[:, i, :].unsqueeze(1)  # [src_length, 1, encoder_hidden_dim * 2]
            root_node = BeamNode(y_hat_n=y_hat_i_t, log_prob_path=[0], s_n=s_i_t, pre_node=None)
            beam_nodes = [root_node] * self.beam_size
            output_nodes = []

            # init beam size, drop 1 whenever a beam meets <eos>
            K = self.beam_size

            # scores for stored beams
            scores_topk = torch.zeros(1, self.beam_size).to(self.device)

            for t in range(1, MAX_LENGTH):
                # start from 1 as the first column are zeros that represent <sos>
                # each time using current y_t, attention, and previous s_{t-1}
                # to compute s_t and predict y_{t+1}_hat
                # we use the same subscript t for y and s here because y starts from 1, s starts from 0

                # explore beam-size * vocab-size possibilities
                y_hat_i_t_full = torch.zeros(1, self.trg_vocab_size * K).to(self.device)
                if isinstance(s_i_t, tuple):
                    s_i_t_full = (torch.zeros(1, self.hidden_dim * K).to(self.device),
                                  torch.zeros(1, self.hidden_dim * K).to(self.device))
                else:
                    s_i_t_full = torch.zeros(1, self.hidden_dim * K).to(self.device)

                for j in range(len(beam_nodes)):
                    node = beam_nodes[j]
                    y_hat_i_t_j, s_i_t_j, _ = self.feed_forward_decoder(node.y_hat_n, node.s_n,
                                                                        encoder_outputs_i, mask[i])
                    # partition a vocab-size range to the current y_hat_i_t_j and s_i_t_j
                    y_hat_i_t_full[:, j * self.trg_vocab_size: (j + 1) * self.trg_vocab_size] = y_hat_i_t_j
                    if isinstance(s_i_t_full, tuple):
                        s_i_t_full[0][:, j * self.hidden_dim: (j + 1) * self.hidden_dim] = s_i_t_j[0]
                        s_i_t_full[1][:, j * self.hidden_dim: (j + 1) * self.hidden_dim] = s_i_t_j[1]
                    else:
                        s_i_t_full[:, j * self.hidden_dim: (j + 1) * self.hidden_dim] = s_i_t_j

                # avoid repeating for time-step 1
                if t == 1:
                    # mask out all but the first beam (expanded from <sos>)
                    inds_except_first_beam = range(self.trg_vocab_size, self.trg_vocab_size * self.beam_size)
                    if self.beam_size > 1:
                        y_hat_i_t_full.index_fill_(dim=1,
                                                   index=torch.tensor(inds_except_first_beam).to(self.device),
                                                   value=-float("Inf"))

                # Example to explain the following expansion
                # a = [[1, 2, 3]] => a.t() = [[1], [2], [3]] => a.expand(-1, v) = [[1]*v, [2]*v, [3]*v]
                # => reshape(1, v) = [[1..., 2..., 3...]] of size (1, 3 * v),
                # here beam = 3, trg-vocab-size = v

                scores_t = scores_topk.t().expand(-1, self.trg_vocab_size) \
                    .reshape(1, K * self.trg_vocab_size)  # scores from previous time-step

                scores_topk, indices = torch.topk(y_hat_i_t_full + scores_t,
                                                  dim=1, k=K)  # [1, beam_size]

                prev_node_inds = [ind // self.trg_vocab_size for ind in indices[0]]  # know which node belongs to

                if t == 1:
                    assert prev_node_inds == [0] * self.beam_size

                new_beam_nodes = []
                kept_beam_inds = []

                for k in range(K):
                    y_hat_n = (indices[0, k] % self.trg_vocab_size).unsqueeze(0)  # fix the index, torch.Size([1])
                    prev_node_ind = prev_node_inds[k]
                    prev_node = beam_nodes[prev_node_ind]
                    if isinstance(s_i_t_full, tuple):
                        s_n = (s_i_t_full[0][:, prev_node_ind * self.hidden_dim: (prev_node_ind + 1) * self.hidden_dim],
                               s_i_t_full[1][:, prev_node_ind * self.hidden_dim: (prev_node_ind + 1) * self.hidden_dim])
                    else:
                        s_n = s_i_t_full[:, prev_node_ind * self.hidden_dim: (prev_node_ind + 1) * self.hidden_dim]

                    if y_hat_n == self.eos_idx:
                        output_nodes.append(BeamNode(y_hat_n=y_hat_n,
                                                     s_n=s_n,
                                                     log_prob_path=prev_node.log_prob_path + [scores_topk[:, k]],
                                                     pre_node=prev_node))
                    else:
                        kept_beam_inds.append(k)
                        new_beam_nodes.append(BeamNode(y_hat_n=y_hat_n,
                                                       s_n=s_n,
                                                       log_prob_path=prev_node.log_prob_path + [scores_topk[:, k]],
                                                       pre_node=prev_node))
                beam_nodes = new_beam_nodes
                scores_topk = scores_topk.index_select(dim=1, index=torch.tensor(kept_beam_inds, dtype=torch.long)
                                                       .to(self.device))
                K = len(beam_nodes)

            # backtrace
            max_log_prob = -float('inf')
            end_node = None
            n = 0
            # B for biased score, N for normal score, O for openNMT's implementation
            for node in output_nodes:
                if self.score_choice == "B":
                    # biased to earlier tokens N*score_1 + (N-1)*score_2 + ... 1*score_N
                    normalised_log_prob_n = sum(node.log_prob_path) / (len(node.log_prob_path) ** self.length_norm_ratio)
                elif self.score_choice == "N":
                    normalised_log_prob_n = node.log_prob_path[-1] / (len(node.log_prob_path) ** self.length_norm_ratio)
                elif self.score_choice == "O+B":
                    denominator = ((5 + len(node.log_prob_path)) ** self.length_norm_ratio) / \
                                  ((5 + 1) ** self.length_norm_ratio)
                    normalised_log_prob_n = sum(node.log_prob_path) / denominator
                elif self.score_choice == "O+N":
                    denominator = ((5 + len(node.log_prob_path)) ** self.length_norm_ratio) / \
                                  ((5 + 1) ** self.length_norm_ratio)
                    normalised_log_prob_n = node.log_prob_path[-1] / denominator
                else:
                    normalised_log_prob_n = 0
                    print("No such scoring choice!")
                if normalised_log_prob_n > max_log_prob:
                    end_node = node
                    max_log_prob = normalised_log_prob_n
                n += 1

            T = MAX_LENGTH
            path = []
            while end_node is not None:
                path.append(end_node)
                end_node = end_node.pre_node
            path.reverse()
            for t in range(len(path)):
                decoded_batch[i, t] = path[t].y_hat_n[0]
            for t in range(len(path), T):
                decoded_batch[i, t] = self.eos_idx

        return decoded_batch
