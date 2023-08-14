import json
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd


def read_lm_kbc_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """
    Reads a LM-KBC jsonl file and returns a list of dictionaries.
    Args:
        file_path: path to the jsonl file
    Returns:
        list of dictionaries, each possibly has the following keys:
        - "SubjectEntity": str
        - "Relation": str
        - "ObjectEntities":
            None or List[List[str]] (can be omitted for the test input)
    """
    rows = []
    with open(file_path, "r") as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)
    return rows


def read_lm_kbc_jsonl_to_df(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Reads a LM-KBC jsonl file and returns a dataframe.
    """
    rows = read_lm_kbc_jsonl(file_path)
    df = pd.DataFrame(rows)
    return df


def save_df_to_jsonl(file_path: Union[str, Path], df: Union[pd.DataFrame, List]):
    """
    Saves the dataframe into a jsonl file.
    # TODO: not workable on DataFrame
    """
    with open(file_path, "w") as f:
        for result in df:
            f.write(json.dumps(result) + "\n")


def read_prompt_template_to_dict(file_path):
    """
    Reads a propmt template csv file and returns a dict.
    :param file_path:
    :return:
    """
    prompt_template = pd.read_csv(file_path)
    return dict(zip(prompt_template.loc[:, 'Relation'], prompt_template.loc[:, 'PromptTemplate']))


def concat_jsonl(dir_path, output_path):
    """

    :param dir_path: path to the folder of jsonl files
    :param output_path: path to the concatenated output
    :return:
    """
    concat_results = []
    for path in Path(dir_path).glob('*.jsonl'):
        with open(path, 'r') as f:
            results = [json.loads(line) for line in f]
            concat_results += results
    save_df_to_jsonl(output_path, concat_results)
    print(len(concat_results))
    return concat_results
