import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from metric import Correct_Rate, Accuracy, Align, insertions, deletions, substitutions

def get_op(seq1, seq2):
    return ['D' if s1 != "<eps>" and s2 == "<eps>" else
            'I' if s1 == "<eps>" and s2 != "<eps>" else
            'S' if s1 != s2 else 'C' for s1, s2 in zip(seq1, seq2)]

def get_align(k, s1, s2):
    a1, a2 = Align(s1, s2)
    I = insertions(a1, a2)
    D = deletions(a1, a2)[0]
    S = substitutions(a1, a2)
    C = len(a1) - I - D - S

    return [
        k + ' ref ' + ' '.join(a1),
        k + ' hyp ' + ' '.join(a2),
        k + ' op ' + ' '.join(get_op(a1, a2)),
        k + f' #csid {C} {S} {I} {D}'
    ]

def evaluate_from_dfs(truth_df: pd.DataFrame,
                      pred_df: pd.DataFrame,
                      output_dir: str = 'aligned',
                      wov: bool = False,
                      print_output: bool = True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_col = 'Reference_vow_phn' if wov else 'Reference_phn'
    ann_col = 'Annotation_vow_phn' if wov else 'Annotation_phn'

    valid_ids = set(truth_df['ID']) & set(pred_df['ID'])
    truth_df = truth_df[truth_df['ID'].isin(valid_ids)][['ID', ref_col, ann_col]]
    pred_df = pred_df[pred_df['ID'].isin(valid_ids)][['ID', 'Prediction']]

    # Merge the necessary columns only
    df = pd.merge(truth_df, pred_df, on='ID')

    canon_pred_f = open(output_dir / 'ref_our_detail', 'w')
    canon_trans_f = open(output_dir / 'ref_human_detail', 'w')
    trans_pred_f = open(output_dir / 'human_our_detail', 'w')

    list_correct_rate, list_accuracy, list_len = [], [], []

    for _, row in df.iterrows():
        data_id = row['ID']
        pred = row['Prediction'].split()
        trans = row[ann_col].split()
        canon = row[ref_col].split()

        canon_pred_f.write('\n'.join(get_align(data_id, canon, pred)) + '\n')
        canon_trans_f.write('\n'.join(get_align(data_id, canon, trans)) + '\n')
        trans_pred_f.write('\n'.join(get_align(data_id, trans, pred)) + '\n')

        correct_rate, len_, _ = Correct_Rate(trans, pred)
        acc, len_ = Accuracy(trans, pred)

        list_correct_rate.append((len_ - correct_rate) / len_)
        list_accuracy.append((len_ - acc) / len_)
        list_len.append(len_)

    canon_pred_f.close()
    canon_trans_f.close()
    trans_pred_f.close()

    weights = np.array(list_len)
    corr_rate = round(np.sum(np.array(list_correct_rate) * weights) / weights.sum(), 4)
    acc = round(np.sum(np.array(list_accuracy) * weights) / weights.sum(), 4)

    if print_output:
        print("** MD&D Evaluation **")
        print("Correct Rate:", corr_rate)
        print("Accuracy:", acc)

    return corr_rate, acc

def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction accuracy against annotated and canonical phonemes.")
    parser.add_argument('--truth-file', type=str, required=True, help="Path to CSV containing truth data (with canonical and annotated phonemes)")
    parser.add_argument('--pred-file', type=str, required=True, help="Path to CSV containing predictions (with ID and Prediction columns)")
    parser.add_argument('--output-dir', type=str, default='./aligned', help="Directory to save alignment output files")
    parser.add_argument('--wov', action='store_true', help="Use vowel-only phoneme columns instead of full ones")

    args = parser.parse_args()

    # Load data
    truth_df = pd.read_csv(args.truth_file)
    pred_df = pd.read_csv(args.pred_file)

    # Run evaluation
    evaluate_from_dfs(
        truth_df=truth_df,
        pred_df=pred_df,
        output_dir=args.output_dir,
        wov=args.wov
    )

if __name__ == '__main__':
    main()
