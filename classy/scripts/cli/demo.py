from argparse import ArgumentParser

from classy.scripts.cli.utils import get_device
from classy.utils.commons import execute_bash_command


def populate_parser(parser: ArgumentParser):
    parser.add_argument("model_path")
    parser.add_argument("-p", "--port", type=int, default=8000)
    parser.add_argument("-d", "--device", default="gpu")


def get_parser(subparser=None) -> ArgumentParser:
    # subparser: Optional[argparse._SubParsersAction]

    parser_kwargs = dict(name="demo", description="expose a demo of a classy model with Streamlit", help="TODO")
    parser = (subparser.add_parser if subparser is not None else ArgumentParser)(**parser_kwargs)

    populate_parser(parser)

    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    # import here to avoid importing before needed
    import sys
    from streamlit.cli import main as st_main

    print(sys.argv)
    device = get_device(args.device)

    sys.argv = [
        "streamlit",
        "run",
        "classy/scripts/model/demo.py",
        args.model_path,
        str(device),
        "--server.port",
        str(args.port),
    ]

    sys.exit(st_main())


if __name__ == "__main__":
    main(parse_args())
