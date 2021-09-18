from argparse import ArgumentParser

from classy.scripts.cli.utils import get_device, autocomplete_model_path


def populate_parser(parser: ArgumentParser):
    parser.add_argument("model_path").completer = autocomplete_model_path
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

    device = get_device(args.device)

    # script params
    script_params = [args.model_path]
    if device != -1:
        # todo ugly workaround for straemlit which interprets -1 as a streamlit param)
        script_params.append(str(device))

    sys.argv = [
        "streamlit",
        "run",
        "classy/scripts/model/demo.py",
        *script_params,
        "--server.port",
        str(args.port),
    ]

    sys.exit(st_main())


if __name__ == "__main__":
    main(parse_args())
