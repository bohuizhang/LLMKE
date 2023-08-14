import sys
from collections import Counter, defaultdict
from SPARQLWrapper import SPARQLWrapper, JSON

from pipeline.file_io import read_lm_kbc_jsonl_to_df, save_df_to_jsonl


def sparql_query(query):
    """ regular sparql query with formatting functions

    :param query:
    :return:
    """
    endpoint_url = "https://query.wikidata.org/sparql"
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6

    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql_results = sparql.query().convert()

    object_entities, object_entities_id = list(), list()
    for result in sparql_results["results"]["bindings"]:
        object_entities_id.append(result['object']['value'].strip('http://www.wikidata.org/entity/'))
        object_entities.append(result['objectLabel']['value'])

    if not object_entities:
        return [""], [""]

    return object_entities_id, object_entities


def query_answers(subject_entity_id, relation):
    """

    :param subject_entity_id:
    :param relation:
    :return:
    """
    if relation == 'BandHasMember':  # train: 5, val: 5
        q = """
            SELECT DISTINCT ?object ?objectLabel
            WHERE
            {
              ?object wdt:P463 wd:%s .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
            }
            """ % subject_entity_id
    elif relation == 'CityLocatedAtRiver':  # (check) train: 37
        q = """
            SELECT DISTINCT ?object ?objectLabel
            WHERE
            {
              wd:%s wdt:P206 ?object .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
            }
            """ % subject_entity_id
    elif relation == 'CompanyHasParentOrganisation':  # train: 2
        q = """
            SELECT DISTINCT ?object ?objectLabel 
            WHERE
            {
              wd:%s wdt:P749 ?object .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
            }
            """ % subject_entity_id
    elif relation == 'CompoundHasParts':  # (check) train: 33
        q = """
            SELECT DISTINCT ?object ?objectLabel 
            WHERE
            {
              wd:%s wdt:P527 ?object .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
            }
            """ % subject_entity_id
    elif relation == 'CountryBordersCountry':  # train: 9, val: 12
        q = """
            SELECT DISTINCT ?object ?objectLabel 
            WHERE
            {
              wd:%s wdt:P47 ?object .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
            }
            """ % subject_entity_id
    elif relation == 'CountryHasOfficialLanguage':  # train: 1, val: 1
        q = """
            SELECT DISTINCT ?object ?objectLabel 
            WHERE
            {
              wd:%s wdt:P37 ?object .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
            }
            """ % subject_entity_id
    elif relation == 'CountryHasStates':
        q = """
            SELECT DISTINCT ?object ?objectLabel 
            WHERE
            {
              wd:%s wdt:P150 ?object .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
            }
            """ % subject_entity_id
    elif relation == 'FootballerPlaysPosition':  # val: 1
        q = """
            SELECT DISTINCT ?object ?objectLabel 
            WHERE
            {
              wd:%s wdt:P413 ?object .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
            }
            """ % subject_entity_id
    elif relation == 'PersonCauseOfDeath':  # val: 1
        q = """
            SELECT DISTINCT ?object ?objectLabel 
            WHERE
            {
              wd:%s wdt:P509 ?object .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
            }
            """ % subject_entity_id
    elif relation == 'PersonHasAutobiography':  # train: 5, val: 5
        q = """
            SELECT DISTINCT ?object ?objectLabel 
            WHERE
            {
              { 
                ?object wdt:P50 wd:%s ;
                        wdt:P136 wd:Q4184 .
              }
              UNION
              {
                ?object wdt:P50 wd:%s ;
                        wdt:P136 wd:Q112983 .
              }
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
            }
            """ % (subject_entity_id, subject_entity_id)
    elif relation == 'PersonHasEmployer':  # train: 1, val: 4
        q = """
            SELECT DISTINCT ?object ?objectLabel 
            WHERE
            {
              wd:%s wdt:P108 ?object .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
            }
            """ % subject_entity_id
    elif relation == 'PersonHasNoblePrize':
        q = """
            SELECT DISTINCT ?object ?objectLabel 
            WHERE
            {
              wd:%s wdt:P166 ?object .
              wd:Q7191 wdt:P527 ?object .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
            }
            """ % subject_entity_id
    elif relation == 'PersonHasNumberOfChildren':  # train: 2, val: 1
        q = """
            SELECT DISTINCT ?object ?objectLabel 
            WHERE
            {
              wd:%s wdt:P1971 ?object .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
            }
            """ % subject_entity_id
    elif relation == 'PersonHasPlaceOfDeath':  # val: 1
        q = """
            SELECT DISTINCT ?object ?objectLabel 
            WHERE
            {
              wd:%s wdt:P20 ?object .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
            }
            """ % subject_entity_id
    elif relation == 'PersonHasProfession':  # train: 4, val: 2
        q = """
            SELECT DISTINCT ?object ?objectLabel 
            WHERE
            {
              wd:%s wdt:P106 ?object .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
            }
            """ % subject_entity_id
    elif relation == 'PersonHasSpouse':  # val: 1
        q = """
            SELECT DISTINCT ?object ?objectLabel 
            WHERE
            {
              wd:%s wdt:P26 ?object .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
            }
            """ % subject_entity_id
    elif relation == 'PersonPlaysInstrument':  # train: 1, val: 1
        q = """
            SELECT DISTINCT ?object ?objectLabel 
            WHERE
            {
              wd:%s wdt:P1303 ?object .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
            }
            """ % subject_entity_id
    elif relation == 'PersonSpeaksLanguage':  # val: 2
        q = """
            SELECT DISTINCT ?object ?objectLabel 
            WHERE
            {
              wd:%s wdt:P1412 ?object .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
            }
            """ % subject_entity_id
    elif relation == 'RiverBasinsCountry':
        q = """
            SELECT DISTINCT ?object ?objectLabel 
            WHERE
            {
              wd:%s wdt:P205 ?object .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
            }
            """ % subject_entity_id
    elif relation == 'SeriesHasNumberOfEpisodes':  # val: 1
        q = """
            SELECT DISTINCT ?object ?objectLabel 
            WHERE
            {
              wd:%s wdt:P1113 ?object .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
            }
            """ % subject_entity_id
    elif relation == 'StateBordersState':  # train: 6, val: 4
        q = """
            SELECT DISTINCT ?object ?objectLabel 
            WHERE
            {
              wd:%s wdt:P47 ?object .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
            }
            """ % subject_entity_id
    else:
        print("Not supported!")
        return None
    return sparql_query(q)


def check_completeness(data_df):
    """

    :param data_df:
    :return:
    """
    relations = data_df['Relation'].unique()

    unmatched_cases_count = defaultdict(int)
    for relation in relations:
        relation_df = data_df[data_df['Relation'] == relation]
        relation_gt = dict(zip(relation_df['SubjectEntityID'], relation_df['ObjectEntitiesID']))

        number_of_unmatched_cases = 0
        for key, value in relation_gt.items():
            queried_qids, _ = query_answers(key, relation)
            if Counter(queried_qids) == Counter(value):
                pass
            else:
                # print("Subject:", key)
                # print("ObjectEntitiesID:", value)
                # print("Query:", queried_qids)
                # print("Differences:", set(queried_qids) - set(value))
                number_of_unmatched_cases += 1
        unmatched_cases_count[relation] = number_of_unmatched_cases

    return unmatched_cases_count


def query_test_objects(test_data, output_path):
    """

    :param test_data:
    :param output_path:
    :return:
    """
    results = []
    for idx, row in test_data.iterrows():
        object_entities_id, object_entities = query_answers(row['SubjectEntityID'], row['Relation'])
        result = {
            "SubjectEntityID": row["SubjectEntityID"],
            "SubjectEntity": row["SubjectEntity"],
            "Relation": row["Relation"],
            "ObjectEntities": object_entities,
            "ObjectEntitiesID": object_entities_id
        }
        results.append(result)
    save_df_to_jsonl(output_path, results)
    # return results


if __name__ == "__main__":
    # train_df = read_lm_kbc_jsonl_to_df("../data/train.jsonl")
    # print(check_completeness(train_df))

    # val_df = read_lm_kbc_jsonl_to_df("../data/val.jsonl")
    # print(check_completeness(val_df))

    test_df = read_lm_kbc_jsonl_to_df("../data/test.jsonl")
    query_test_objects(test_df, "../data/test.query.jsonl")
