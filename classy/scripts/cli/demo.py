from argparse import ArgumentParser

from classy.scripts.cli.utils import get_device, autocomplete_model_path, checkpoint_path_from_user_input


def populate_parser(parser: ArgumentParser):
    parser.add_argument("model_path", type=checkpoint_path_from_user_input).completer = autocomplete_model_path
    parser.add_argument("-p", "--port", type=int, default=8000)
    parser.add_argument("-d", "--device", default="gpu")
    parser.add_argument("--prediction-params", type=str, default=None, help="Path to prediction params")


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
        script_params += ["cuda_device", str(device)]
    if args.prediction_params is not None:
        script_params += ["prediction_params", args.prediction_params]

    sys.argv = [
        "streamlit",
        "run",
        # __file__ points to this file's location, even when pip installed.
        # given our code structure (this file is [...]/classy/scripts/cli/demo.py),
        # if we replace /cli/ with /model/ we get the actual streamlit python file we need to run.
        __file__.replace("/cli/", "/model/"),
        *script_params,
        "--server.port",
        str(args.port),
    ]

    sys.exit(st_main())


if __name__ == "__main__":
    main(parse_args())
