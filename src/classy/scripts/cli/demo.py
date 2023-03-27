import functools
from argparse import ArgumentParser

from ...utils.help_cli import (
    HELP_DRY_MODEL_CONFIGURATION,
    HELP_MODEL_PATH,
    HELP_PREDICTION_PARAMS,
)
from ...utils.optional_deps import requires
from .utils import autocomplete_model_path, checkpoint_path_from_user_input, get_device


def populate_parser(parser: ArgumentParser):
    parser.add_argument(
        "model_path",
        type=functools.partial(checkpoint_path_from_user_input, include_dry_model=True),
        help=HELP_MODEL_PATH,
    ).completer = functools.partial(autocomplete_model_path, include_dry_model=True)
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="The port where the streamlit demo will be exposed.",
    )
    parser.add_argument(
        "-d",
        "--device",
        default=None,
        help="On which device the model for the demo will be loaded. If not provided, classy will try to infer the desired behavior from the available environment.",
    )
    parser.add_argument(
        "--prediction-params", type=str, default=None, help=HELP_PREDICTION_PARAMS
    )
    parser.add_argument(
        "--dry-model-configuration",
        type=str,
        default=None,
        help=HELP_DRY_MODEL_CONFIGURATION,
    )


def get_parser(subparser=None) -> ArgumentParser:
    # subparser: Optional[argparse._SubParsersAction]

    parser_kwargs = dict(description="expose a demo of a classy model with Streamlit")
    if subparser is not None:
        parser_kwargs["name"] = "demo"
        parser_kwargs["help"] = "Expose a demo of a classy model with Streamlit."
    else:
        parser_kwargs["prog"] = "demo"

    parser = (subparser.add_parser if subparser is not None else ArgumentParser)(
        **parser_kwargs
    )

    populate_parser(parser)

    return parser


def parse_args():
    return get_parser().parse_args()


@requires("streamlit", "demo")
def main(args):
    # import here to avoid importing before needed
    import sys

    from streamlit.web.cli import main as st_main

    device = get_device(args.device)

    # script params
    script_params = [args.model_path]
    if device is not None and device != -1:
        # todo ugly workaround for streamlit which interprets -1 as a streamlit param)
        script_params += ["cuda_device", str(device)]
    if args.prediction_params is not None:
        script_params += ["prediction_params", args.prediction_params]
    if args.dry_model_configuration is not None:
        script_params += ["dry_model_configuration", args.dry_model_configuration]

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
