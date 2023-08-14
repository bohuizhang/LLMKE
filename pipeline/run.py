import json
import os
from collections import defaultdict

from pipeline.disambiguate import disambiguation_baseline
from pipeline.models import get_chat_completion_response
from pipeline.context import wikipedia_geographical_context_loader, imdb_context_loader, wikipedia_context_loader
from pipeline.file_io import read_lm_kbc_jsonl_to_df, save_df_to_jsonl
from pipeline.prompt import example_generator, prompt_generator


def run(args):
    """

    :param args:
    :return:
    """
    print('Starting probing: {}, {}, {}, {}, {}\n'.format(
        args.dataset, args.model, args.setting, args.prompt, args.relation
    ))

    # load data
    data_df = read_lm_kbc_jsonl_to_df('data/{}.jsonl'.format(args.dataset))
    data_df = data_df[data_df['Relation'] == args.relation].copy()

    # load prompt template
    if args.prompt == 'question':
        with open('../question-prompts.json', 'r') as f:
            prompt_templates = json.load(f)
        prompt_template = prompt_templates[args.relation]
    elif args.prompt == 'triple':
        prompt_template = args.relation
    else:
        raise NotImplementedError("Prompt type -p shoud be within ['question', 'triple'].")

    # build few-shot examples
    # n_examples = 5
    if args.setting == 'few-shot':  # or 'context':
        train_df = read_lm_kbc_jsonl_to_df('data/train.jsonl')
        examples = example_generator(args.prompt, train_df, args.relation, prompt_template, False)
    elif args.setting == 'sem-sim' or 'context':  # PersonHasNobelPrize excluded
        if os.path.exists('../examples.jsonl'):
            example_df = read_lm_kbc_jsonl_to_df('../examples.jsonl')
            examples = example_generator(args.prompt, example_df, args.relation, prompt_template, True)
        else:
            raise FileNotFoundError('No semantic similar examples found!')
    else:
        examples = defaultdict(str)
    print(examples, '\n')

    # prompting
    results = []
    for idx, row in data_df[:].iterrows():
        query = prompt_generator(args.prompt, row['SubjectEntity'], prompt_template)
        query += " Format the response as a Python list such as '[\"answer_a\", \"answer_b\"]'."
        print("{} {}".format(idx, query))
        if args.setting == 'zero-shot':
            result = {
                "SubjectEntityID": row["SubjectEntityID"],
                "SubjectEntity": row["SubjectEntity"],
                "Relation": row["Relation"],
                "ObjectEntities": get_chat_completion_response(args.model, query, args.relation),
                "ObjectEntitiesID": []
            }
        elif args.setting == 'few-shot' or 'sem-sim':
            result = {
                "SubjectEntityID": row["SubjectEntityID"],
                "SubjectEntity": row["SubjectEntity"],
                "Relation": row["Relation"],
                "ObjectEntities": get_chat_completion_response(args.model, query, args.relation, examples),
                "ObjectEntitiesID": []
            }
        elif args.setting == 'context':
            # if context is empty it will follow the few-shot setting
            if args.relation == 'CountryHasStates':
                context = wikipedia_geographical_context_loader(row['SubjectEntity'])
            elif args.relation == 'SeriesHasNumberOfEpisodes':
                context = imdb_context_loader(row['SubjectEntity'], id_index_path='../context/imdb.series.index.json')
            else:
                context = wikipedia_context_loader(row['SubjectEntity'])
            result = {
                "SubjectEntityID": row["SubjectEntityID"],
                "SubjectEntity": row["SubjectEntity"],
                "Relation": row["Relation"],
                "ObjectEntities": get_chat_completion_response(args.model, query, args.relation, examples, context),
                "ObjectEntitiesID": []
            }
        else:
            raise NotImplementedError("Setting `-s` should be within ['few-shot', 'zero-shot', 'context']")
        # special treatment of numeric relations, do not execute disambiguation
        if result["Relation"] == "PersonHasNumberOfChildren" or result["Relation"] == "SeriesHasNumberOfEpisodes":
            result["ObjectEntitiesID"] = [str(item) for item in result["ObjectEntities"]]
        # normal relations: execute Wikidata's disambiguation
        else:
            for s in result['ObjectEntities']:
                result["ObjectEntitiesID"].append(disambiguation_baseline(str(s)))

        results.append(result)

    # save results
    predictions_dir = 'predictions/{}-{}-{}-{}'.format(args.dataset, args.model, args.setting, args.prompt)
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
    save_df_to_jsonl(os.path.join(predictions_dir, '{}.jsonl'.format(args.relation)), results)

    print('Finished probing: {}, {}, {}, {}, {}'.format(
        args.dataset, args.model, args.setting, args.prompt, args.relation
    ))
