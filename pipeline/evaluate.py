import argparse
import os.path

import seaborn as sns
import matplotlib.pylab as plt

from collections import Counter
from prettytable import PrettyTable
from pipeline.file_io import *


def true_positives(preds: List, gts: List) -> int:
    tp = 0
    for pred in preds:
        if pred in gts:
            tp += 1

    return tp


def precision(preds: List[str], gts: List[str]) -> float:
    # when nothing is predicted, precision 1 irrespective of the ground truth value
    try:
        if len(preds) == 0:
            return 1
        # When the predictions are not empty
        return min(true_positives(preds, gts) / len(preds), 1.0)
    except TypeError:
        return 0.0


def recall(preds: List[str], gts: List[str]) -> float:
    try:
        # When ground truth is empty return 1 even if there are predictions (edge case)
        if len(gts) == 0 or gts == [""]:
            return 1.0
        # When the ground truth is not empty
        return true_positives(preds, gts) / len(gts)
    except TypeError:
        return 0.0


def f1_score(p: float, r: float) -> float:
    try:
        return (2 * p * r) / (p + r)
    except ZeroDivisionError:
        return 0.0


def rows_to_dict(rows: List[Dict]) -> Dict:
    return {(r["SubjectEntity"], r["Relation"]): r["ObjectEntitiesID"] for r in rows}


def evaluate_per_sr_pair(pred_rows, gt_rows) -> List[Dict[str, float]]:
    pred_dict = rows_to_dict(pred_rows)
    gt_dict = rows_to_dict(gt_rows)

    results = []

    for subj, rel in gt_dict:
        # get the ground truth objects
        gts = gt_dict[(subj, rel)]

        # get the predictions
        preds = pred_dict[(subj, rel)]

        # calculate the scores
        p = precision(preds, gts)
        r = recall(preds, gts)
        f1 = f1_score(p, r)

        results.append({
            "SubjectEntity": subj,
            "Relation": rel,
            "p": p,
            "r": r,
            "f1": f1
        })

        # if p > 1.0 or r > 1.0:
        #     print(f"{subj} {rel} {p} {r} {f1} {gts} {preds}")

    return sorted(results, key=lambda x: (x["Relation"], x["SubjectEntity"]))


def combine_scores_per_relation(scores_per_sr: List[Dict[str, float]]) -> dict:
    scores = {}
    for r in scores_per_sr:
        if r["Relation"] not in scores:
            scores[r["Relation"]] = []
        scores[r["Relation"]].append({
            "p": r["p"],
            "r": r["r"],
            "f1": r["f1"],
        })

    for rel in scores:
        scores[rel] = {
            "p": sum([x["p"] for x in scores[rel]]) / len(scores[rel]),
            "r": sum([x["r"] for x in scores[rel]]) / len(scores[rel]),
            "f1": sum([x["f1"] for x in scores[rel]]) / len(scores[rel]),
        }

    return scores


def display_unmatched_records(pred_rows, gt_rows):
    """

    :param pred_rows:
    :param gt_rows:
    :returns:
    """
    pred_df = pd.DataFrame(pred_rows).rename(columns={"ObjectEntities": "Predictions",
                                                      "ObjectEntitiesID": "PredictionsID"})
    gt_df = pd.DataFrame(gt_rows).rename(columns={"ObjectEntities": "GroundTruths",
                                                  "ObjectEntitiesID": "GroundTruthsID"})
    records_df = pd.merge(gt_df, pred_df, on=["SubjectEntityID", "SubjectEntity"])

    # entities in ground truths but not in predictions
    records_df['MissingEntities'] = records_df.apply(
        lambda x: set(x['GroundTruths']).difference(set(x['Predictions'])), axis=1)
    records_df['MissingEntitiesID'] = records_df.apply(
        lambda x: set(x['GroundTruthsID']).difference(set(x['PredictionsID'])), axis=1)

    # entities in predictions but not in ground truths
    records_df['ErrorEntities'] = records_df.apply(
        lambda x: set(x['Predictions']).difference(set(x['GroundTruths'])), axis=1)
    records_df['ErrorEntitiesID'] = records_df.apply(
        lambda x: set(x['PredictionsID']).difference(set(x['GroundTruthsID'])), axis=1)

    # entities with correct labels but wrong IDs
    records_df['Ambiguities'] = records_df.apply(
        lambda x: True if len(set(x['PredictionsID']) & set(x['GroundTruthsID'])) < len(
            set(x['Predictions']) & set(x['GroundTruths'])) else False, axis=1)

    unmatched_records_table = PrettyTable()
    unmatched_records_table.field_names = [
        "Subject entity ID", "Subject entity",
        "Predictions", "Ground truths",
        "Predictions ID", "Ground truths ID",
        "Missing entities", "Missing entities ID",
        "Error entities", "Error entities ID",
        "Ambiguities"
    ]

    for idx, row in records_df.iterrows():
        if Counter(row['PredictionsID']) != Counter(row['GroundTruthsID']):
            unmatched_records_table.add_row([
                row["SubjectEntityID"], row["SubjectEntity"],
                row["Predictions"], row["GroundTruths"],
                row["PredictionsID"], row["GroundTruthsID"],
                row["MissingEntities"], row["MissingEntitiesID"],
                row["ErrorEntities"], row["ErrorEntitiesID"],
                row["Ambiguities"]
            ])

    return unmatched_records_table


def extract_errors(pred_rows: List[Dict], gt_rows: List[Dict], relation: str = None):
    """

    :param pred_rows:
    :param gt_rows:
    :param relation:
    :returns:
    """
    pred_df = pd.DataFrame(pred_rows).rename(columns={"ObjectEntities": "Predictions",
                                                      "ObjectEntitiesID": "PredictionsID"})
    gt_df = pd.DataFrame(gt_rows).rename(columns={"ObjectEntities": "GroundTruths",
                                                  "ObjectEntitiesID": "GroundTruthsID"})
    if relation:
        gt_df = gt_df[gt_df['Relation'] == relation]
    records_df = pd.merge(gt_df, pred_df, on=["SubjectEntityID", "SubjectEntity", "Relation"])

    errors = []
    for idx, row in records_df.iterrows():
        if Counter(row['PredictionsID']) != Counter(row['GroundTruthsID']):
            errors.append(row)
    errors_df = pd.DataFrame(errors)
    return errors_df


def sample_errors(errors_df, n_samples):
    """

    :param errors_df:
    :param n_samples:
    :return:
    """
    return errors_df.sample(n_samples, random_state=1)


def evaluate(args):
    """

    :param args:
    :return:
    """
    # paths
    pred_path = 'predictions/{}-{}-{}-{}/{}.jsonl'.format(
        args.dataset, args.model, args.setting, args.prompt, args.relation
    )

    if args.relation == 'all' and not os.path.exists(pred_path):
        pred_rows = concat_jsonl(
            dir_path='predictions/{}-{}-{}-{}'.format(args.dataset, args.model, args.setting, args.prompt),
            output_path='predictions/{}-{}-{}-{}.jsonl'.format(args.dataset, args.model, args.setting, args.prompt)
        )
    else:
        pred_rows = read_lm_kbc_jsonl(pred_path)
    if args.dataset == 'test':
        gt_rows = read_lm_kbc_jsonl('data/test.query.jsonl')
    else:
        gt_rows = read_lm_kbc_jsonl('data/{}.jsonl'.format(args.dataset))
    if args.relation != 'all':
        gt_rows = [row for row in gt_rows if row['Relation'] == args.relation]

    scores_per_sr_pair = evaluate_per_sr_pair(pred_rows, gt_rows)
    scores_per_relation = combine_scores_per_relation(scores_per_sr_pair)

    scores_per_relation["*** Average ***"] = {
        "p": sum([x["p"] for x in scores_per_relation.values()]) / len(scores_per_relation),
        "r": sum([x["r"] for x in scores_per_relation.values()]) / len(scores_per_relation),
        "f1": sum([x["f1"] for x in scores_per_relation.values()]) / len(scores_per_relation),
    }

    results = pd.DataFrame(scores_per_relation).transpose().round(3)
    print(results)

    # display unmatched records
    if args.compare:
        unmatched_records = display_unmatched_records(pred_rows, gt_rows)
        if args.write:
            save_dir = 'evaluations/{}-{}-{}-{}'.format(args.dataset, args.model, args.setting, args.prompt)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if args.relation == 'all':
                save_path = 'evaluations/{}-{}-{}-{}.txt'.format(args.dataset, args.model, args.setting, args.prompt)
            else:
                save_path = os.path.join(save_dir, '{}.txt'.format(args.relation))
            with open(save_path, 'w+', encoding='utf-8') as f:
                f.write(results.to_string())
                f.write('\n')
                f.write(unmatched_records.get_string())
            print('Evaluation results saved at {}.'.format(save_path))
        else:
            print(unmatched_records)
    elif args.write:
        save_dir = 'evaluations/{}-{}-{}-{}'.format(args.dataset, args.model, args.setting, args.prompt)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if args.relation == 'all':
            save_path = 'evaluations/{}-{}-{}-{}.txt'.format(args.dataset, args.model, args.setting, args.prompt)
        else:
            save_path = os.path.join(save_dir, '{}.txt'.format(args.relation))
        with open(save_path, 'w+', encoding='utf-8') as f:
            f.write(results.to_string())
        print('Evaluation results saved at {}.'.format(save_path))

    # visualize results using heatmap
    if args.relation == 'all':
        sns.set(rc={'figure.figsize': (20, 12)})
        sns.set(font_scale=1.75)
        ax = sns.heatmap(results, vmin=0.0, vmax=1.0, cmap="YlGnBu", annot=True, fmt=".2f")
        ax.set_title('data: {}, model: {}, setting: {}'.format(args.dataset, args.model, args.setting), y=1.025)
        plt.subplots_adjust(left=0.25, right=1.025, top=0.9, bottom=0.075)
        plt.savefig('evaluations/{}-{}-{}-{}.png'.format(args.dataset, args.model, args.setting, args.prompt))
        # plt.show()
