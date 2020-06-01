import pandas as pd
import numpy as np
from Levenshtein import distance


class AlternatingCharacterTable:

    def __init__(self, act_path):
        self.act = pd.read_excel(act_path, encoding='UTF-8')
        self.act['alternative_characters'] = self.act['alternative_characters'].apply(lambda x: x.split(','))
        self.act = self.act['alternative_characters']

    @staticmethod
    def text_to_df(path_pred, path_ref):
        with open(path_pred, 'r', encoding='UTF-8-sig') as f:
            lines_pred = f.readlines()
        with open(path_ref, 'r', encoding='UTF-8-sig') as f:
            lines_ref = f.readlines()
        df = pd.DataFrame(columns=['PRED', 'REF'])
        df['PRED'] = lines_pred
        df['ref'] = lines_ref
        df['PRED'] = df['PRED'].apply(lambda w: w.replace('\n', '').replace(' ', ''))
        df['REF'] = df['REF'].apply(lambda w: w.replace('\n', '').replace(' ', ''))
        return df

    @staticmethod
    def tsv_to_df(tsv_path):
        return pd.read_csv(tsv_path, sep='\t', header=0)

    def compute_ACC_ACT(self, df):
        # calculate MED
        print(df)
        dist = df.apply(lambda x: distance(str(x['PRED']), str(x['REF'])), axis=1)

        # zero MED means correct
        correct = np.sum(dist == 0)
        n = len(df)
        acc = correct / n

        # extract those with MED 1 or 2 and look up the ACT
        for_act = df[(dist > 0) & (dist <= 2)]
        # print('The number of pred/ref pairs with MED of 1 or 2 is', len(for_act))
        correct_act = np.sum(for_act.apply(lambda x: self.look_up_ACT(str(x['PRED']), str(x['REF'])), axis=1))
        # print('The number of replaceable names is', correct_act)
        acc_act = (correct + correct_act) / n

        return {
            'acc': acc,
            'acc-act': acc_act,
            'replaced': str(correct_act) + '/' + str(len(for_act))
        }

    def look_up_ACT(self, pred, ref):

        # Method that examines if pred and ref are equivalent
        # after looking up Alternating Character Table.
        # Check of the MED is done before this method.

        # The primary assumption requires pred and ref to be of the same length
        if not len(pred) == len(ref):
            return False

        for i in range(len(pred)):
            # every time find two distinct characters at the same position, check the table
            if not pred[i] == ref[i]:
                replaceable = any(self.act.apply(lambda x: (pred[i] in x) and (ref[i] in x)))
                if not replaceable:
                    return False

        # all the distinct characters are 'replaceable' in ACT
        return True
