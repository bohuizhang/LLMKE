# LLMKE
Code for ISWC LM-KBC 2023 Challenge from Team KCL.

## Files

```text
.
├── context
│   └── imdb.series.index.json
├── data
│   ├── dev.pred.jsonl
│   ├── test.jsonl
│   ├── test.query.jsonl  # Query date: 28/07/2023
│   ├── train.jsonl
│   └── val.jsonl
├── evaluations           # Disambiguated
│   └── */*.txt
├── predictions           # Disambiguated 
│   └── */*.jsonl
├── pipeline
│   ├── __init__.py
│   ├── config.py
│   ├── disambiguate.py
│   ├── evaluate.py
│   ├── context.py
│   ├── file_io.py
│   ├── models.py
│   ├── prompt.py
│   └── run.py
├── examples.jsonl
├── main.py
├── predictions.jsonl
├── predictions.zip
├── question-prompts.json
├── README.md 
└── sparql_query.py 
```
For detailed results, please refer to the spreadsheet [here](https://docs.google.com/spreadsheets/d/1hIaJ96g8K0lLvlS2CP5wszSWQg54ms7hFvWodfipAOQ/edit?usp=sharing).

## Run

You need an OpenAI API key to run this pipeline. You can paste your API key into `pipeline.config.py`.

```shell
cd lm-kbc-2023
```

```shell
python main.py -t run -d <dataset> -m <model> -s <setting> -p <prompt> -r <relation>
```
  - `<dataset>`: `train`, `val`, `test` 
  - `<model>`: `gpt-3.5-turbo`, `gpt-4` 
  - `<setting>`: `zero-shot`, `few-shot`, `context`
  - `<prompt>`: `question`, `triple`
  - e.g. `python main.py -t run -d test -m gpt-4 -s few-shot -p question -r CompoundHasParts`

For using IMDb context, run `download_imdb_dataset()` and `build_imdb_id_index()` in `pipeline.context` first. 
We provide an index for the test set.

## Disambiguate:

```shell
python main.py -t disambiguate -d <dataset> -m <model> -s <setting> -p <prompt> -r <relation>
```
  - e.g. `python main.py -t disambiguate -d test -m gpt-4 -s context -p question -r StateBordersState`

## Evaluate:
- A single relation: 
```shell
python main.py -t evaluate -d <dataset> -m <model> -s <setting> -p <prompt> -c -w -r <relation>
```

- The whole set: 
```shell
python main.py -t evaluate -d <dataset> -m <model> -s <setting> -p <prompt> -w -r all
```

## TODO
- Prompting:
  - Improve prompts: self-critique, majority vote, etc.
  - Semantic similar few-shot examples
- Wikidata Qnode disambiguator
  - Relations need to be systematically improved: **BandHasMember**, **StateBordersState** 
  - Relations with several cases: CityLocatedAtRiver, CountryHasOfficialLanguage, PersonHasAutobiography, PersonHasSpouse
