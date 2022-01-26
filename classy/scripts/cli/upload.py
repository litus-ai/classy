from argparse import ArgumentParser


def populate_parser(parser: ArgumentParser):
    parser.add_argument("model_name", help="The model you want to upload.")
    parser.add_argument(
        "--organization",
        help="The name of the organization where you want to upload the model.",
    )
    parser.add_argument(
        "--name",
        help="Optional name to use when uploading to the HuggingFace repository.",
    )
    parser.add_argument(
        "--commit", help="Commit message to use when pushing to the HuggingFace Hub."
    )


def get_parser(subparser=None) -> ArgumentParser:
    parser_kwargs = dict(
        name="upload",
        description="upload a pretrained model to your (or an organization's) HuggingFace Hub",
        help="Upload a pretrained model to your (or an organization's) HuggingFace Hub.",
    )
    parser = (subparser.add_parser if subparser is not None else ArgumentParser)(
        **parser_kwargs
    )

    populate_parser(parser)

    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    from classy.scripts.model.upload import upload

    upload(args.model_name, args.organization, args.name, args.commit)


if __name__ == "__main__":
    main(parse_args())
