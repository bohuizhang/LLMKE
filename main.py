import argparse

import openai

from pipeline.config import OPENAI_API_KEY
from pipeline.disambiguate import disambiguate
from pipeline.evaluate import evaluate
from pipeline.run import run


def main():
    parser = argparse.ArgumentParser(
        description="Arguments of input for task functions"
    )

    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default="run",
        choices=["run", "evaluate", "disambiguate"],
        required=True,
        help="Task"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="val",
        required=True,
        help="Dataset"
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gpt-3.5-turbo",
        type=str,
        required=True,
        help="Model"
    )
    parser.add_argument(
        "-s",
        "--setting",
        default="few-shot",
        type=str,
        choices=['zero-shot', 'few-shot', 'context', 'sem-sim'],
        required=True,
        help="Probing setting"
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default='question',
        choices=['question', 'triple', 'submission'],
        required=True,
        help="Prompt type"
    )
    parser.add_argument(
        "-r",
        "--relation",
        type=str,
        required=True,
        help="Relation"
    )
    parser.add_argument(
        "-c",
        "--compare",
        action="store_true",
        required=False,
        help="Display the difference if True when evaluation (optional)"
    )
    parser.add_argument(
        "-w",
        "--write",
        action="store_true",
        required=False,
        help="Path if save the evaluation results (optional)"
    )

    args = parser.parse_args()

    openai.api_key = OPENAI_API_KEY

    if args.task == "run":
        run(args)
    elif args.task == "evaluate":
        evaluate(args)
    elif args.task == "disambiguate":
        disambiguate(args)
    else:
        raise NotImplementedError("Please select your task from ['run', 'evaluate', 'disambiguate'].")


if __name__ == "__main__":
    main()
