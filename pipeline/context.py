import gzip
import json
import os.path
import re
from typing import Dict, Union

import pandas as pd
import requests

from collections import Counter, defaultdict
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

import wget
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError

from pipeline.file_io import read_lm_kbc_jsonl_to_df


def wikipedia_paragraphs_loader(entity: str, sections: list) -> str:
    """ load the Wikipedia paragraphs of an entity and section

    :param entity: entity string
    :param sections: list of section titles
    :return: Wikipedia content string
    """
    wikipedia.set_lang("en")
    try:
        page_content = wikipedia.page(entity, auto_suggest=False).content
        # page_content = wikipedia.page(entity).content
    # except DisambiguationError as e:
    #     print("Disambiguation occurs for '{}'".format(entity))
    #     random_choice = random.choice(e.options)
    #     page_content = wikipedia.page(random_choice).content
    #     # page_content = wikipedia.page(entity, auto_suggest=True).content
    except (DisambiguationError, PageError):
        return ""
    wikipedia_paragraphs = defaultdict(str)
    section_content = page_content.split("\n\n\n")
    for section in sections:
        if section == 'introduction':
            wikipedia_paragraphs['introduction'] = section_content[0]
        else:
            # TODO: parse other sections
            pass
    return str(wikipedia_paragraphs)


def wikipedia_infobox_loader(entity: str) -> Union[str, Dict]:
    """ infobox parser

    :param entity:
    :return:
    """
    url = f"https://en.wikipedia.org/wiki/{entity}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    try:
        infobox_table = soup.find('table', {'class': re.compile('infobox.*')}).find_all("tr")
    except AttributeError:
        return ""
    infobox_dict = dict()
    for row in infobox_table:
        try:
            infobox_dict[row.th.get_text()] = row.td.get_text(separator=', ', strip=True)
        except AttributeError:
            pass
    return infobox_dict


def wikipedia_geographical_context_loader(entity):
    wikipedia.set_lang("en")
    entity = 'Administrative divisions of {}'.format(entity)
    try:
        page_content = wikipedia.page(entity, auto_suggest=False).content
        return page_content[:32000]  # control page length to not exceed 4k-tokens limit
    except (DisambiguationError, PageError):
        return ""


def wikipedia_context_loader(entity):
    introduction = wikipedia_paragraphs_loader(entity, ['introduction'])
    infobox = wikipedia_infobox_loader(entity)
    if introduction and infobox:
        return "Wikipedia introduction: \"{}\"; Wikipedia Infobox: {}.".format(introduction, infobox)
    elif introduction:
        return "Wikipedia introduction: \"{}\".".format(introduction)
    elif infobox:
        return "Wikipedia Infobox: {}".format(infobox)
    else:
        return ""


def wikidata_infobox_knowledge_gap(data_df, relation, infobox_relations, integer_only=False):
    """

    :param data_df:
    :param relation:
    :param infobox_relations:
    :param integer_only:
    :return:
    """
    relation_df = data_df[data_df['Relation'] == relation]
    inconsistency_count, consistency_count = 0, 0
    for idx, row in relation_df.iterrows():
        infobox_dict = wikipedia_infobox_loader(row['SubjectEntity'])
        if not infobox_dict:
            infobox_dict = dict()
        # else:
        #     print(infobox_dict)
        wikipedia_objects_full_list = list()
        for infobox_relation in infobox_relations:
            try:
                wikipedia_objects = re.sub(r'\[.*?]', '', infobox_dict[infobox_relation]).lower()
                wikipedia_objects = re.sub(r'\(.*?\)', '', wikipedia_objects).strip()
                if integer_only and re.search(r'\d+', wikipedia_objects):
                    wikipedia_objects = re.sub("[^0-9]", "", wikipedia_objects)
                wikipedia_objects = wikipedia_objects.strip(',')
                wikipedia_objects = list(map(lambda x: x.strip(), wikipedia_objects.split(',')))
                if integer_only and not wikipedia_objects[0].isdigit():
                    wikipedia_objects = list(str(len(wikipedia_objects)))
                # remove emopty & noisy strings in the list
                # wikipedia_objects = list(filter(None, wikipedia_objects))
                wikipedia_objects = [entity for entity in wikipedia_objects if entity not in ['', '\u200b']]
                # TODO: map entity strings to Wikidata QIDs
                wikipedia_objects_full_list += wikipedia_objects
            except KeyError:
                pass
        wikidata_objects = [obj.lower() for obj in row['ObjectEntities']]
        if not wikipedia_objects_full_list:
            wikipedia_objects_full_list = ['']
        if Counter(wikidata_objects) != Counter(wikipedia_objects_full_list):
            print(row['SubjectEntity'], row['SubjectEntityID'], wikidata_objects, wikipedia_objects_full_list)
            inconsistency_count += 1
        else:
            consistency_count += 1
    return inconsistency_count, consistency_count


def download_imdb_dataset(dataset='title.basics.tsv.gz', context_path='../context'):
    if os.path.exists(os.path.join(context_path, dataset)):
        print('IMDb dataset {} has been downloaded to {}.'.format(dataset, os.path.join(context_path, dataset)))
        return
    if not os.path.exists(context_path):
        os.makedirs(context_path)
    imdb_dataset_url = 'https://datasets.imdbws.com/{}'.format(dataset)
    file_name = wget.download(imdb_dataset_url, out=context_path)
    print('IMDb dataset {} has been downloaded to {}.'.format(dataset, file_name))
    return


def build_imdb_id_index(data_path, id_index_path):
    data = []
    with gzip.open(data_path) as f:
        for idx, line in enumerate(f.readlines()):
            row = line.decode('UTF-8').strip().split('\t')
            data.append(row)
    data_df = pd.DataFrame(data, columns=data[0])
    series_df = data_df[data_df['titleType'].isin(['tvSeries', 'tvMiniSeries'])]
    id_index = defaultdict(list)
    for idx, row in series_df.iterrows():
        id_index[row['primaryTitle']].append(row['tconst'])
        id_index[row['originalTitle']].append(row['tconst'])
    # deduplicate
    id_index = {key: list(set(value)) for key, value in id_index.items()}
    with open(id_index_path, 'w') as fp:
        json.dump(id_index, fp, sort_keys=True, indent=4)


def imdb_context_loader(entity, id_index_path):
    if not os.path.exists(id_index_path):
        raise FileNotFoundError('Please build the IMDb id index first!')
    with open(id_index_path, 'r') as fp:
        id_index = json.load(fp)
    try:
        imdb_ids = id_index[entity]
    except KeyError:
        return ""
    imdb_urls = ['https://www.imdb.com/title/{}/'.format(imdb_id) for imdb_id in imdb_ids]
    context = list()
    for imdb_url in imdb_urls:
        try:
            request = Request(imdb_url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urlopen(request).read()
            soup = BeautifulSoup(response.decode('utf-8'), 'html.parser')
            title = soup.title.string.strip(' - IMDb')
            episode_number = soup.find('span', {'class': 'ipc-title__subtext'}).string
            context.append('According to IMDB, {} has {} episodes.'.format(title, episode_number))
        except AttributeError:
            pass
    wikipedia_context = wikipedia_context_loader(entity)
    # TODO: refine context selection, instead of using the first one
    if context and wikipedia_context:
        return '{}; {}'.format(str(context[0]), wikipedia_context)
    elif context:
        return str(context[0])
    elif wikipedia_context:
        return wikipedia_context
    else:
        return ""


def knowledge_gap_count(data_path, relation=None):
    """

    :param data_path:
    :param relation:
    :return:
    """
    data_df = read_lm_kbc_jsonl_to_df(data_path)
    wikipedia_relation_mapping = {
        'BandHasMember': ['Members', 'Past members'],
        'CityLocatedAtRiver': None,
        'CompanyHasParentOrganisation': ['Parent', 'Parent company'],
        'CompoundHasParts': None,
        'CountryBordersCountry': None,
        'CountryHasOfficialLanguage': ['Official\xa0languages'],
        'CountryHasStates': None,
        'FootballerPlaysPosition': ['Position(s)'],
        'PersonCauseOfDeath': None,
        'PersonHasAutobiography': None,
        'PersonHasEmployer': ['Employer', 'Employers', 'Employer(s)', 'Institutions'],
        'PersonHasNoblePrize': None,
        'PersonHasNumberOfChildren': ['Children'],  # integer_only=True
        'PersonHasPlaceOfDeath': None,  # 'Died'
        'PersonHasProfession': ['Occupation', 'Occupations', 'Occupation(s)'],
        'PersonHasSpouse': ['Spouse', 'Spouses', 'Spouse(s)'],
        'PersonPlaysInstrument': ['Instruments', 'Instrument(s)'],
        'PersonSpeaksLanguage': None,
        'RiverBasinsCountry': ['Country'],
        'SeriesHasNumberOfEpisodes': ['No. of episodes'],
        'StateBordersState': None
    }
    if relation:
        inconsistency, consistency = wikidata_infobox_knowledge_gap(data_df, relation,
                                                                    wikipedia_relation_mapping[relation])
        print('Relation: {}, Inconsistency: {}, Consistency: {}'.format(relation, inconsistency, consistency))
        return {relation: {'inconsistency': inconsistency, 'consistency': consistency}}
    else:
        results = defaultdict(dict)
        for relation, wikipedia_relations in wikipedia_relation_mapping.items():
            if wikipedia_relations:
                inconsistency, consistency = wikidata_infobox_knowledge_gap(data_df, relation, wikipedia_relations)
                print('Relation: {}, Inconsistency: {}, Consistency: {}'.format(relation, inconsistency, consistency))
                results[relation] = {'inconsistency': inconsistency, 'consistency': consistency}
        return results


# if __name__ == "__main__":
#     results = knowledge_gap_count('../data/test.query.jsonl', 'CompanyHasParentOrganisation')
