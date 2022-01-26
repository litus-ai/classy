from argparse import ArgumentParser

from argcomplete import FilesCompleter

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
    # TODO: would be cool to have it work with exp_name and add an optional --checkpoint-name flag (default=best.ckpt)
    # the user should not need to know what a checkpoint is :)

    subcmd = parser.add_subparsers(
        dest="subcmd",
        required=True,
        help="Whether you want to use the model interactively or to process a file.",
    )
    interactive_parser = subcmd.add_parser("interactive")
    interactive_parser.add_argument(
        "model_path", type=checkpoint_path_from_user_input, help=HELP_MODEL_PATH
    ).completer = autocomplete_model_path
    interactive_parser.add_argument(
        "-d",
        "--device",
        default="gpu",
        help="The device where the dataset prediction will be run.",
    )
    interactive_parser.add_argument(
        "--prediction-params", type=str, default=None, help="Path to prediction params."
    )

    file_parser = subcmd.add_parser("file")
    file_parser.add_argument(
        "model_path", type=checkpoint_path_from_user_input, help=HELP_MODEL_PATH
    ).completer = autocomplete_model_path
    file_parser.add_argument(
        "file_path", help="The file containing the instances that you want to process."
    ).completer = FilesCompleter()
    file_parser.add_argument(
        "-d",
        "--device",
        default="gpu",
        help="The device you will use for the prediction.",
    )
    file_parser.add_argument(
        "-o",
        "--output-path",
        required=True,
        help="The file where the predictions will be stored.",
    ).completer = FilesCompleter()
    file_parser.add_argument(
        "--prediction-params", type=str, default=None, help=HELP_PREDICTION_PARAMS
    )
    file_parser.add_argument(
        "--token-batch-size", type=int, default=1024, help=HELP_TOKEN_BATCH_SIZE
    )


def get_parser(subparser=None) -> ArgumentParser:
    # subparser: Optional[argparse._SubParsersAction]

    parser_kwargs = dict(
        name="predict",
        description="predict with a model trained using classy",
        help="Predict with a model trained using classy.",
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
    from classy.scripts.model.predict import file_main, interactive_main

    subcmd = args.subcmd

    device = get_device(args.device)

    if subcmd == "file":
        file_main(
            args.model_path,
            args.file_path,
            args.output_path,
            args.prediction_params,
            device,
            args.token_batch_size,
        )
    elif subcmd == "interactive":
        interactive_main(args.model_path, args.prediction_params, device)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main(parse_args())
