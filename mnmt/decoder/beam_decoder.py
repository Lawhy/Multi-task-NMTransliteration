from mnmt.decoder import BasicDecoder
import random
import torch


class BeamNode:
    def __init__(self, y_hat_n, log_prob_n, s_n, pre_node, y_hat_path):
        self.y_hat_n = y_hat_n
        self.log_prob_n = log_prob_n
        self.s_n = s_n
        self.pre_node = pre_node
        self.y_hat_path = y_hat_path


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
        y_hat = self.init_decoder_outputs(trg)  # [trg_length, batch_size, trg_vocab_size (input_dim)]
        s_t = self.init_s_0(encoder_final_state)

        y_hat_t = trg[0, :]  # first input to the decoder is the <sos> tokens

        # decode each sample in the batch
        for i in range(batch_size):
            y_hat_i_t = trg[0, :].unsqueeze(0)  # [1, trg_vocab_size]
            print(y_hat_i_t, y_hat_i_t.shape)

            if isinstance(s_t, tuple):
                s_i_t = (s_t[0][i].unsqueeze(0), s_t[1][i].unsqueeze(0))  # [1, hidden_dim], tuple
            else:
                s_i_t = s_t[i].unsqueeze(0)  # [1, hidden_dim]

            encoder_outputs_i = encoder_outputs[:, i, :].unsqueeze(1)  # [src_length, 1, encoder_hidden_dim * 2]
            root_node = BeamNode(y_hat_n=y_hat_i_t, log_prob_n=0, s_n=s_i_t, pre_node=None, y_hat_path=y_hat[:, i, :])
            #  y_hat_path = [trg_length, trg_vocab_size]
            batch_nodes = [root_node]
            print(root_node.y_hat_n.shape, root_node.s_n[0].shape, root_node.y_hat_path.shape)
            print("-----------")

            for t in range(1, trg.size(0)):
                # start from 1 as the first column are zeros that represent <sos>
                # each time using current y_t, attention, and previous s_{t-1}
                # to compute s_t and predict y_{t+1}_hat
                # we use the same subscript t for y and s here because y starts from 1, s starts from 0

                # explore beam-size * vocab-size possibilities
                y_hat_i_t_full = torch.zeros(1, self.trg_vocab_size * self.beam_size).to(self.device)
                if isinstance(s_i_t, tuple):
                    s_i_t_full = (torch.zeros(1, self.hidden_dim * self.beam_size).to(self.device),
                                  torch.zeros(1, self.hidden_dim * self.beam_size).to(self.device))
                else:
                    s_i_t_full = torch.zeros(1, self.hidden_dim * self.beam_size).to(self.device)

                print(y_hat_i_t_full.shape, s_i_t_full[0].shape)

                teacher_force = random.random() < teacher_forcing_ratio
                for j in range(len(batch_nodes)):
                    node = batch_nodes[j]
                    y_hat_i_t_j, s_i_t_j, _ = self.feed_forward_decoder(node.y_hat_n, node.s_n, encoder_outputs_i, mask)
                    # partition a vocab-size range to the current y_hat_i_t_j and s_i_t_j
                    y_hat_i_t_full[:, j*self.trg_vocab_size: (j + 1)*self.trg_vocab_size] = y_hat_i_t_j
                    if isinstance(s_i_t_full, tuple):
                        s_i_t_full[0][:, j * self.hidden_dim: (j + 1) * self.hidden_dim] = s_i_t_j[0]
                        s_i_t_full[1][:, j * self.hidden_dim: (j + 1) * self.hidden_dim] = s_i_t_j[1]
                    else:
                        s_i_t_full[:, j * self.hidden_dim: (j + 1) * self.hidden_dim] = s_i_t_j

                y_hat_i_t_topk, indices = torch.topk(y_hat_i_t_full, dim=1, k=self.beam_size)  # [1, beam_size]
                prev_node_inds = [ind // self.trg_vocab_size for ind in indices[0]]
                new_batch_nodes = []
                for k in range(self.beam_size):
                    y_hat_n = indices[0, k] if not teacher_force else trg[t, i]
                    prev_node_ind = prev_node_inds[k]
                    prev_node = batch_nodes[prev_node_ind]
                    if isinstance(s_i_t_full, tuple):
                        s_n = (s_i_t_full[0][:, prev_node_ind * self.hidden_dim: (prev_node_ind + 1) * self.hidden_dim],
                               s_i_t_full[1][:, prev_node_ind * self.hidden_dim: (prev_node_ind + 1) * self.hidden_dim])
                    else:
                        s_n = s_i_t_full[:, prev_node_ind * self.hidden_dim: (prev_node_ind + 1) * self.hidden_dim]
                    y_hat_path = prev_node.y_hat_path
                    y_hat_path[t, :] = \
                        y_hat_i_t_full[1, prev_node_ind*self.trg_vocab_size: (prev_node_ind + 1)*self.trg_vocab_size]
                    new_batch_nodes.append(BeamNode(y_hat_n=y_hat_n,
                                                    s_n=s_n,
                                                    log_prob_n=prev_node.log_prob_n + 0
                                                    if teacher_force else prev_node.log_prob_n + y_hat_i_t_topk[0, k],
                                                    pre_node=prev_node, y_hat_path=y_hat_path))
                batch_nodes = new_batch_nodes

            # backtrace
            max_log_prob = -float('inf')
            end_node = None
            for node in batch_nodes:
                if node.log_prob_n > max_log_prob:
                    end_node = node
            y_hat[:, i, :] = end_node.y_hat_path

        return y_hat
