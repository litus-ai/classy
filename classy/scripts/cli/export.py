from argparse import ArgumentParser


def populate_parser(parser: ArgumentParser):
    parser.add_argument("model_name", help="The model you want to export")
    parser.add_argument(
        "--zip-name",
        help="Name of the output file. Defaults to classy-export-{model_name}.zip",
    )
    parser.add_argument(
        "-ns",
        "--no-strip",
        action="store_true",
        default=False,
        help="Whether to strip the checkpoint of optimizer states, schedulers and callbacks. "
        "Should only do this if you're not planning on resuming training (i.e., for inference).",
    )
    parser.add_argument(
        "-a",
        "--all-ckpts",
        action="store_true",
        default=False,
        help="Whether to include every checkpoint under the <RUN>/checkpoints/ folder or just the `best.ckpt`.",
    )


def get_parser(subparser=None) -> ArgumentParser:
    parser_kwargs = dict(
        name="export",
        description="export a trained model as a zip file",
        help="Export a trained model as a zip file",
    )
    parser = (subparser.add_parser if subparser is not None else ArgumentParser)(
        **parser_kwargs
    )

    populate_parser(parser)

    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    from classy.scripts.model.export import export

    export(args.model_name, args.no_strip, args.all_ckpts, args.zip_name)


if __name__ == "__main__":
    main(parse_args())
