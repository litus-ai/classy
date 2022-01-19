from argparse import ArgumentParser

from classy.scripts.cli.utils import (
    autocomplete_model_path,
    checkpoint_path_from_user_input,
    get_device,
)
from classy.utils.help_cli import (
    HELP_MODEL_PATH,
    HELP_PREDICTION_PARAMS,
    HELP_TOKEN_BATCH_SIZE,
)


def populate_parser(parser: ArgumentParser):
    parser.add_argument(
        "model_path", type=checkpoint_path_from_user_input, help=HELP_MODEL_PATH
    ).completer = autocomplete_model_path
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="The port where the REST api will be exposed.",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="gpu",
        help="On which device the model for the REST api will be loaded.",
    )
    parser.add_argument(
        "--token-batch-size", type=int, default=1024, help=HELP_TOKEN_BATCH_SIZE
    )
    parser.add_argument(
        "--prediction-params", type=str, default=None, help=HELP_PREDICTION_PARAMS
    )


def get_parser(subparser=None) -> ArgumentParser:
    # subparser: Optional[argparse._SubParsersAction]

    parser_kwargs = dict(
        name="serve",
        description="Expose a model trained with classy on a REST API",
        help="Expose a model trained with classy on a REST API.",
    )
    parser = (subparser.add_parser if subparser is not None else ArgumentParser)(
        **parser_kwargs
    )

    populate_parser(parser)

    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    # import here to avoid importing torch before it's actually needed
    from classy.scripts.model.serve import serve

    device = get_device(args.device)
    serve(
        args.model_path,
        args.port,
        device,
        args.token_batch_size,
        prediction_params=args.prediction_params,
    )


if __name__ == "__main__":
    main(parse_args())
