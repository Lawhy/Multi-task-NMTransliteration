class BasicTranslator:

    def __init__(self, quiet_translate=True):
        """
        Args:
            quiet_translate:  determine whether to print translation or not
        """
        self.quiet_translate = quiet_translate

    def translate(self, output, trg, trg_field, beam_size, output_file=None):
        """
        Args:
            output:  [trg_length, batch_size, output_dim], model's output
            trg:  [trg_length, batch_size], target reference
            trg_field:  target language field
            beam_size:  size for beam_size search
            output_file:  save translation results in output_file
        """
        raise NotImplementedError

    @staticmethod
    def matching(pred, ref, trg_field, quiet_translate, output_file=None):
        """
        Args:
            pred:  model's prediction, modified from model's output
            ref:  target reference, modified from raw target reference {trg}
            trg_field:  target language field
            quiet_translate:  determine whether to print translation or not
            output_file:  save translation results in output_file
        """
        tally = 0
        for j in range(pred.shape[0]):

            pred_j = pred[j, :]
            pred_j_toks = []
            for t in pred_j:
                tok = trg_field.vocab.itos[t]
                if tok == '<eos>':
                    break
                else:
                    pred_j_toks.append(tok)
            pred_j = ''.join(pred_j_toks)

            ref_j = ref[j, :]
            ref_j_toks = []
            for t in ref_j:
                tok = trg_field.vocab.itos[t]
                if tok == '<eos>':
                    break
                else:
                    ref_j_toks.append(tok)
            ref_j = ''.join(ref_j_toks)

            if not quiet_translate:
                print("Pred: {} | Ref: {}".format(pred_j, ref_j))
            if output_file is not None:
                output_file.write(pred_j + '\t' + ref_j + '\n')  # save output results in file

            if pred_j == ref_j:
                tally += 1
        return tally
