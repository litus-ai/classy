from argparse import ArgumentParser
from pathlib import Path

from argcomplete import FilesCompleter

from classy.scripts.cli.utils import (
    autocomplete_model_path,
    checkpoint_path_from_user_input,
    get_device,
)
from classy.utils.help_cli import (
    HELP_EVALUATE,
    HELP_FILE_PATH,
    HELP_MODEL_PATH,
    HELP_PREDICTION_PARAMS,
    HELP_TOKEN_BATCH_SIZE,
)


def populate_parser(parser: ArgumentParser):
    parser.add_argument(
        "model_path",
        type=checkpoint_path_from_user_input,
        help=HELP_MODEL_PATH,
    ).completer = autocomplete_model_path
    parser.add_argument(
        "file_path",
        nargs="?",
        default=None,
        help=HELP_FILE_PATH,
    ).completer = FilesCompleter()
    parser.add_argument(
        "-d",
        "--device",
        default="gpu",
        help="The device you will use for the evaluation.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        default=None,
        required=False,
        help="""
        Optional. If specified, the predictions will be stored in the "output_path" along with the gold labels and the
        original sample.
        """,
    ).completer = FilesCompleter()
    parser.add_argument(
        "--token-batch-size", type=int, default=1024, help=HELP_TOKEN_BATCH_SIZE
    )
    parser.add_argument(
        "--evaluation-config", type=str, default=None, help=HELP_EVALUATE
    )
    parser.add_argument(
        "--prediction-params", type=str, default=None, help=HELP_PREDICTION_PARAMS
    )


def get_parser(subparser=None) -> ArgumentParser:
    # subparser: Optional[argparse._SubParsersAction]

    parser_kwargs = dict(
        name="evaluate",
        description="evaluate a model trained using classy",
        help="Evaluate a model trained using classy.",
    )
    parser = (subparser.add_parser if subparser is not None else ArgumentParser)(
        **parser_kwargs
    )

    populate_parser(parser)

    return parser


def parse_args():
    return get_parser().parse_args()


def automatically_infer_test_path(model_path: str) -> str:
    from classy.utils.lightning import load_training_conf_from_checkpoint

    checkpoint_path = Path(model_path)
    exp_split_data_folder = checkpoint_path.parent.parent.joinpath("data")

    # search if it was created via split at training time
    if exp_split_data_folder.exists():
        possible_test_files = [
            fp for fp in exp_split_data_folder.iterdir() if fp.stem == "test"
        ]
        if len(possible_test_files) == 1:
            return str(possible_test_files[0])

    # check if dataset_path provided at training time was a folder that contained a test set
    training_conf = load_training_conf_from_checkpoint(model_path)
    dataset_path = Path(training_conf.data.datamodule.dataset_path)
    if dataset_path.exists() and dataset_path.is_dir():
        possible_test_files = [fp for fp in dataset_path.iterdir() if fp.stem == "test"]
        if len(possible_test_files) == 1:
            return str(possible_test_files[0])

    raise ValueError


def main(args):
    # import here to avoid importing torch before it's actually needed
    from classy.scripts.model.evaluate import evaluate

    # input_path: if provided, use default one
    # otherwise, try to infer its positions
    if args.file_path is not None:
        input_path = args.file_path
    else:
        # try to infer path
        try:
            input_path = automatically_infer_test_path(args.model_path)
            print(f"Test path automatically inferred to {input_path}")
        except ValueError:
            print("Failed to automatically infer test path")
            input_path = input("Please, explicitly enter test path: ").strip()

    device = get_device(args.device)
    evaluate(
        args.model_path,
        device,
        args.token_batch_size,
        input_path,
        args.output_path,
        evaluate_config_path=args.evaluation_config,
        prediction_params=args.prediction_params,
        metrics_fn=None,
    )


if __name__ == "__main__":
    main(parse_args())
