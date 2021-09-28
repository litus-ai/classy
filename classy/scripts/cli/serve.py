from argparse import ArgumentParser

from classy.scripts.cli.utils import get_device, autocomplete_model_path, checkpoint_path_from_user_input


def populate_parser(parser: ArgumentParser):
    parser.add_argument("model_path", type=checkpoint_path_from_user_input).completer = autocomplete_model_path
    parser.add_argument("-p", "--port", type=int, default=8000)
    parser.add_argument("-d", "--device", default="gpu")
    parser.add_argument("--token-batch-size", type=int, default=128)
    parser.add_argument("--prediction-params", type=str, default=None, help="Path to prediction params")


def get_parser(subparser=None) -> ArgumentParser:
    # subparser: Optional[argparse._SubParsersAction]

    parser_kwargs = dict(name="serve", description="serve a model trained with classy on a REST API", help="TODO")
    parser = (subparser.add_parser if subparser is not None else ArgumentParser)(**parser_kwargs)

    populate_parser(parser)

    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    # import here to avoid importing torch before it's actually needed
    from classy.scripts.model.serve import serve

    device = get_device(args.device)
    serve(args.model_path, args.port, device, args.token_batch_size, prediction_params=args.prediction_params)


if __name__ == "__main__":
    main(parse_args())
