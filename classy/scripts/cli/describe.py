from argparse import ArgumentParser

from classy.data.data_drivers import GENERATION, QA, SENTENCE_PAIR, SEQUENCE, TOKEN
from classy.utils.help_cli import HELP_TASKS


def populate_parser(parser: ArgumentParser):
    parser.add_argument(
        "task",
        choices=[SEQUENCE, SENTENCE_PAIR, TOKEN, QA, GENERATION],
        help=HELP_TASKS,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="The dataset you want to describe (run statistics on).",
    )
    parser.add_argument(
        "--tokenize",
        default=None,
        help="Indicates the language of the dataset in order to select "
        "the correct tokenizer. Must be a valid language code for "
        "the sacremoses tokenizer url: 'https://github.com/alvations/sacremoses'.",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="The port where the streamlit demo will be exposed.",
    )


def get_parser(subparser=None) -> ArgumentParser:
    parser_kwargs = dict(
        name="describe",
        description="run several statistics on the input dataset and expose them on a streamlit page",
        help="Run several statistics on the input dataset and expose them on a streamlit page.",
    )
    parser = (subparser.add_parser if subparser is not None else ArgumentParser)(
        **parser_kwargs
    )

    populate_parser(parser)

    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    # import here to avoid importing before needed
    import sys

    from streamlit.cli import main as st_main

    # script params
    script_params = [args.task, args.dataset]

    if args.tokenize is not None:
        script_params += [args.tokenize]

    sys.argv = [
        "streamlit",
        "run",
        # see classy/scripts/cli/demo.py for an explanation of this line :)
        __file__.replace("/cli/", "/model/"),
        *script_params,
        "--server.port",
        str(args.port),
    ]

    sys.exit(st_main())


if __name__ == "__main__":
    main(parse_args())
