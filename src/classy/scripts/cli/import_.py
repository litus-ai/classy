from argparse import ArgumentParser


def populate_parser(parser: ArgumentParser):
    parser.add_argument(
        "path", help="Path to the zip file with the model run you want to import"
    )
    parser.add_argument(
        "--exp-dir",
        help="Path to the experiments folder where the exported model should be added. "
        "Optional, automatically inferred if running from a classy project root dir.",
    )


def get_parser(subparser=None) -> ArgumentParser:
    parser_kwargs = dict(
        description="import a previously exported trained model from a zip file"
    )
    if subparser is not None:
        parser_kwargs["name"] = "import"
        parser_kwargs[
            "help"
        ] = "Import a previously exported trained model from a zip file."
    else:
        parser_kwargs["prog"] = "import"

    parser = (subparser.add_parser if subparser is not None else ArgumentParser)(
        **parser_kwargs
    )

    populate_parser(parser)

    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    from ..model.import_ import import_zip

    import_zip(args.path, args.exp_dir)


if __name__ == "__main__":
    main(parse_args())
