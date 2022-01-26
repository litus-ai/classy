from argparse import ArgumentParser


def populate_parser(parser: ArgumentParser):
    parser.add_argument(
        "model_name",
        help="The model you want to download (use user@model for a specific model)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="It will download the model even if you already have it in the "
        "cache. Usually required if you interrupted the previous download.",
    )


def get_parser(subparser=None) -> ArgumentParser:
    parser_kwargs = dict(
        name="download",
        description="download a pretrained model from sunglasses-ai's (or a user's) HuggingFace Hub",
        help="Download a pretrained model from sunglasses-ai's (or a user's) HuggingFace Hub.",
    )
    parser = (subparser.add_parser if subparser is not None else ArgumentParser)(
        **parser_kwargs
    )

    populate_parser(parser)

    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    from classy.scripts.model.download import download

    download(args.model_name, args.force_download)


if __name__ == "__main__":
    main(parse_args())
