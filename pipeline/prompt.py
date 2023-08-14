from collections import defaultdict
from typing import List, Dict
import ast

import pandas as pd

from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from torch import topk

from pipeline.evaluate import extract_errors
from .file_io import read_lm_kbc_jsonl, save_df_to_jsonl


def prompt_generator(prompt_type: str, subject_entity: str, prompt_template: str) -> str:
    """ generate prompts based on input entities and templates
    e.g., data: "Vanillin", "CompoundHasParts"
          template: What are the chemical components of {subject_entity}?
          prompt type: question => What are the chemical components of Vanillin?
          prompt type: triple   => Vanillin, CompoundHasParts

    :param prompt_type: type of prompt from ['question', 'triple']
    :param subject_entity: subject entity string
    :param prompt_template: template string
    :return: prompt
    """
    if prompt_type == 'question':
        return prompt_template.replace('{subject_entity}', subject_entity)
    elif prompt_type == 'triple':
        return '{}, {}:'.format(subject_entity, prompt_template)
    else:
        raise NotImplementedError("`prompt_type` shoud be within ['question', 'triple'].")


def example_generator(
        prompt_type: str, data_df: pd.DataFrame, relation: str, prompt_template: str, similar: bool) -> Dict:
    """

    :param prompt_type:
    :param data_df:
    :param relation:
    :param prompt_template:
    :param similar:
    :return:
    """
    empty_relation_list = [
        "CompanyHasParentOrganisation",
        "PersonCauseOfDeath",
        "PersonHasNoblePrize",
        "PersonHasPlaceOfDeath"
    ]

    examples = defaultdict(str)
    data_df = data_df[data_df['Relation'] == relation].copy()
    if not similar:
        data_df = data_df[2:]
    for idx, row in data_df[:5].iterrows():
        q = prompt_generator(prompt_type, row['SubjectEntity'], prompt_template)
        # use double quotes for strings in the list to avoid SyntaxError
        a = '["' + '", "'.join(row['ObjectEntities']) + '"]'
        examples[q] = a
    if relation in empty_relation_list:  # add the empty example for specific relations
        if '[""]' not in examples.values():
            for idx, row in data_df.iterrows():
                if row['ObjectEntities'] == [""]:
                    q = prompt_generator(prompt_type, row['SubjectEntity'], prompt_template)
                    a = '["' + '", "'.join(row[
                                               'ObjectEntities']) + '"]'
                    examples[q] = a
                    break
    return examples


def verbalise_records(records_df, dataset='corpus'):
    verbalised_records = list()
    for idx, row in records_df.iterrows():
        if dataset == 'corpus':
            verbalised_records.append(
                '{}, {}, {}'.format(row['SubjectEntity'], row['Relation'], str(row['ObjectEntities'])))
        else:
            verbalised_records.append(
                '{}, {}, {}'.format(row['SubjectEntity'], row['Relation'], str(row['GroundTruths'])))
    return verbalised_records


def clustering_records(records: List[str], n_clusters: int) -> List[str]:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(records, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    kmeans_clustering = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(embeddings)
    closest, _ = pairwise_distances_argmin_min(kmeans_clustering.cluster_centers_, embeddings)
    center_records = [records[idx] for idx in closest]
    return center_records


def find_similar_examples_in_corpus(queries: List[str], corpus: List[str]) -> List[str]:
    """

    :param queries:
    :param corpus:
    :return:
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    corpus_embeddings = model.encode(corpus, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    matched_records = list()
    for query in queries:
        error_embedding = model.encode(query)
        cos_scores = util.cos_sim(error_embedding, corpus_embeddings)[0]
        top_embeds = topk(cos_scores, k=1)
        matched_record = corpus[top_embeds[1]]
        matched_records.append(matched_record)
    return matched_records


def sample_few_shot_examples(predictions_dir, output_path, relations=None, n_examples=3):
    """

    :param predictions_dir:
    :param output_path:
    :param relations: list of relations
    :param n_examples:
    :return:
    """
    # load corpus
    train_rows = read_lm_kbc_jsonl('../data/train.jsonl')
    val_rows = read_lm_kbc_jsonl('../data/val.jsonl')
    corpus = train_rows + val_rows
    corpus_df = pd.DataFrame(corpus)
    gt_rows = read_lm_kbc_jsonl('../data/test.query.jsonl')

    # set range of relations
    if not relations:
        relations = list(corpus_df['Relation'].unique())

    # find similar records in corpus based on errors
    few_shot_examples = list()
    for relation in relations:
        # find typical errors
        pred_path = '{}/{}.jsonl'.format(predictions_dir, relation)
        pred_rows = read_lm_kbc_jsonl(pred_path)
        errors_df = extract_errors(pred_rows, gt_rows, relation)
        if not errors_df.empty:
            verbalised_errors = verbalise_records(errors_df, dataset='test')
            typical_errors = clustering_records(verbalised_errors, n_clusters=n_examples)
            # find similar records
            relation_df = corpus_df[corpus_df['Relation'] == relation]
            verbalised_corpus = verbalise_records(relation_df, dataset='corpus')
            matched_records = find_similar_examples_in_corpus(typical_errors, verbalised_corpus)
            # save records
            for r in matched_records:
                subject = r.split(',')[0]
                relation = r.split(',')[1].strip()
                objects = ast.literal_eval(r[r.find('['):])
                few_shot_examples.append({
                    'SubjectEntity': subject,
                    'Relation': relation,
                    'ObjectEntities': objects
                })

    save_df_to_jsonl(output_path, few_shot_examples)
    # return few_shot_examples


# if __name__ == "__main__":
#     sample_few_shot_examples(
#         predictions_dir='../predictions/test-gpt-4-few-shot-question',
#         output_path='../examples.jsonl',
#         n_examples=5
#     )
