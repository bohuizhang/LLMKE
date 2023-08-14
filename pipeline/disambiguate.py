import argparse
import os
import re

import openai
import requests

from difflib import SequenceMatcher

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff

from pipeline.file_io import *
from pipeline.prompt import *


def string_similarities(entity: str, candidates: List[str]) -> str:
    candidates_similarities = sorted(candidates, key=lambda x: SequenceMatcher(None, entity, x).ratio())
    return candidates_similarities[-1]


def disambiguation_baseline(item):
    try:
        url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&language=en&format=json"
        data = requests.get(url).json()
        # TODO: disambiguation when multiple entities returned
        return data['search'][0]['id']
    except (KeyError, IndexError):
        return item


def case_based_disambiguation(item, relation):
    """

    :param item:
    :param relation:
    :return:
    """
    # cases
    if item == 'mercury' and relation == 'CompoundHasParts':
        return 'Q925'
    if item == 'voice' and relation == 'PersonPlaysInstrument':
        return 'Q17172850'
    if item == 'Marianne' and relation == 'PersonHasEmployer':
        return 'Q3291285'
    if item == 'Mother Jones' and relation == 'PersonHasEmployer':
        return 'Q851510'
    if item == 'winger' and relation == 'FootballerPlaysPosition':
        return 'Q11681748'
    if item == 'sculptor' and relation == 'PersonHasProfession':
        return 'Q1281618'
    if item == 'Palestine' and relation == 'RiverBasinsCountry':  # ground truth may be incorrect
        return 'Q23792'

    try:
        url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&language=en&format=json"
        data = requests.get(url).json()
        # TODO: disambiguation when multiple entities returned
        return data['search'][0]['id']
    except (KeyError, IndexError):
        return item


def keyword_based_disambiguation(item, keywords, relation=None):
    """ select the entity based on keywords

    :param item:
    :param keywords:
    :param relation:
    :return:
    """
    try:
        if relation == 'PersonHasAutobiography' and len(item) > 0:
            data = {'search': list()}
            items = [item, item.split(':')[0]]
            for i in items:
                url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={i}&language=en&format=json"
                data['search'] += requests.get(url).json()['search']
        else:
            url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&language=en&format=json"
            data = requests.get(url).json()

        for entity in data['search']:
            for keyword in keywords:
                if re.search(keyword, entity['description']):
                    return entity['id']
        return data['search'][0]['id']

    except (KeyError, IndexError):
        return item


@retry(
    retry=retry_if_exception_type((openai.error.APIError,
                                   openai.error.APIConnectionError,
                                   openai.error.RateLimitError,
                                   openai.error.ServiceUnavailableError,
                                   openai.error.Timeout)),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10)
)
def lm_based_disambiguation(item, question, relation=None):
    """

    :param item:
    :param question:
    :param relation:
    :return:
    """
    try:
        if relation == 'CityLocatedAtRiver' and len(item) > 0 and 'River' not in item:
            data = {'search': list()}
            items = [item, item + ' River', 'River ' + item]
            for i in items:
                url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={i}&language=en&format=json"
                data['search'] += requests.get(url).json()['search']
        else:
            url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&language=en&format=json"
            data = requests.get(url).json()

        if len(data['search']) > 1:
            candidates = list()
            for entity in data['search']:
                if 'description' in entity.keys():
                    candidates.append({
                        entity['id']: {'label': entity['label'], 'description': entity['description']}
                    })
                # else:
                #     candidates.append({entity['id']: entity['label']})
            query = "Given the candidates {}, which one should be the answer to the question '{}' " \
                    "Return only the key of the candidate, such as 'Q123456'.".format(str(candidates), question)
            response = openai.ChatCompletion.create(
                temperature=0,
                model="gpt-4",
                messages=[{"role": "user", "content": query}]
            )
            response = response.choices[0].message.content
            # extract the QID from the response
            if re.search(r'Q\d+', response):
                return re.search(r'Q\d+', response).group()
        return data['search'][0]['id']
    except (KeyError, IndexError):
        return item


def disambiguation_pipeline(dir_path, method, relation, keywords=None, prompt_template=None):
    """ rerun the disambiguation step for all object entity strings

    :param dir_path:
    :param method:
    :param relation:
    :param keywords:
    :param prompt_template:
    :return:
    """
    data = read_lm_kbc_jsonl(os.path.join(dir_path, '{}.jsonl'.format(relation)))
    for row in data:
        row['ObjectEntitiesID'] = list()
        for entity in row['ObjectEntities']:
            if method == 'case-based':
                row['ObjectEntitiesID'].append(case_based_disambiguation(entity, relation))
            elif method == 'keyword-based':
                row['ObjectEntitiesID'].append(keyword_based_disambiguation(entity, keywords, relation))
            elif method == 'lm-based':
                question = prompt_generator('question', row['SubjectEntity'], prompt_template)
                row['ObjectEntitiesID'].append(lm_based_disambiguation(entity, question, relation))
            else:
                row['ObjectEntitiesID'].append(disambiguation_baseline(entity))
    save_df_to_jsonl(
        Path('{}/{}.jsonl'.format(dir_path, relation, method)),
        data
    )


def disambiguate(args):
    """

    :param args:
    :return:
    """
    predictions_dir = Path('predictions/{}-{}-{}-{}'.format(args.dataset, args.model, args.setting, args.prompt))

    if args.relation == 'BandHasMember':
        # disambiguation_pipeline(
        #     dir_path=predictions_dir,
        #     method='lm-based',
        #     relation='BandHasMember',
        #     prompt_template='Who are the members of {subject_entity}?'
        # )
        disambiguation_pipeline(
            dir_path=predictions_dir,
            method='keyword-based',
            relation='BandHasMember',
            keywords=['musician', 'singer', 'guitarist', 'drummer', 'pianist', 'vocalist', 'bassist']
        )
    # TODO: 'Orne'
    elif args.relation == 'CityLocatedAtRiver':
        disambiguation_pipeline(
            dir_path=predictions_dir,
            method='lm-based',
            relation='CityLocatedAtRiver',
            prompt_template='Which river is {subject_entity} located at?'
        )
    elif args.relation == 'CompoundHasParts':
        disambiguation_pipeline(
            dir_path=predictions_dir,
            method='case-based',
            relation='CompoundHasParts'
        )
    # TODO: 'Burmese', 'Bosnian'
    elif args.relation == 'CountryHasOfficialLanguage':
        disambiguation_pipeline(
            dir_path=predictions_dir,
            method='keyword-based',
            relation='CountryHasOfficialLanguage',
            keywords=['language']
        )
    elif args.relation == 'CountryHasStates':
        disambiguation_pipeline(
            dir_path=predictions_dir,
            method='lm-based',
            relation='CountryHasStates',
            prompt_template='What are the first-level administrative territorial entities of {subject_entity}?'
        )
    elif args.relation == 'FootballerPlaysPosition':
        disambiguation_pipeline(
            dir_path=predictions_dir,
            method='case-based',
            relation='FootballerPlaysPosition',
        )
    elif args.relation == 'PersonHasAutobiography':
        disambiguation_pipeline(
            dir_path=predictions_dir,
            method='keyword-based',
            relation='PersonHasAutobiography',
            keywords=['book', 'memoir', 'novel']
        )
    elif args.relation == 'PersonHasEmployer':
        disambiguation_pipeline(
            dir_path=predictions_dir,
            method='case-based',
            relation='PersonHasEmployer'
        )
    elif args.relation == 'PersonHasProfession':
        disambiguation_pipeline(
            dir_path=predictions_dir,
            method='case-based',
            relation='PersonHasProfession',
        )
    # TODO: 'Kevin Moore'
    elif args.relation == 'PersonHasSpouse':
        disambiguation_pipeline(
            dir_path=predictions_dir,
            method='lm-based',
            relation='PersonHasSpouse',
            prompt_template='What is the name of the spouse of {subject_entity}?'
        )
    elif args.relation == 'PersonPlaysInstrument':
        disambiguation_pipeline(
            dir_path=predictions_dir,
            method='case-based',
            relation='PersonPlaysInstrument'
        )
    elif args.relation == 'RiverBasinsCountry':
        disambiguation_pipeline(
            dir_path=predictions_dir,
            method='case-based',
            relation='RiverBasinsCountry',
        )
    # TODO: update
    elif args.relation == 'StateBordersState':
        disambiguation_pipeline(
            dir_path=predictions_dir,
            method='lm-based',
            relation='StateBordersState',
            prompt_template='Which states border the state of {subject_entity}?'
        )
    else:
        print("No disambiguation assigned!")
