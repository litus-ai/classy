from argparse import ArgumentParser
from pathlib import Path


def populate_parser(parser: ArgumentParser):

    # TODO: add help?
    parser.add_argument("task", choices=("sequence", "token", "sentence-pair"))
    parser.add_argument("dataset", type=Path)
    parser.add_argument("-m", "--model-name", default="bert")
    parser.add_argument("-n", "--exp-name", "--experiment-name", dest="exp_name", default=None)
    parser.add_argument("-d", "--device", default="gpu")  # TODO: add validator?
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("-c", "--config", nargs="+", default=[])


def get_parser(subparser=None) -> ArgumentParser:
    # subparser: Optional[argparse._SubParsersAction]

    parser_kwargs = dict(name="train", description="train a model with classy", help="TODO")
    parser = (subparser.add_parser if subparser is not None else ArgumentParser)(**parser_kwargs)

    populate_parser(parser)

    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    # import here to avoid importing torch before it's actually needed
    from hydra import compose, initialize
    from classy.scripts.model.train import train, fix

    if args.root is not None:
        config_name = args.root
    else:
        task = args.task
        config_name = f"{task}-{args.model_name}"

    overrides = []

    # choose device
    device = "cuda" if args.device == "gpu" else args.device
    overrides.append(f"device={device}")

    # create default experiment name if not provided
    exp_name = args.exp_name or f"{args.task}-{args.model_name}"
    overrides.append(f"exp_name={exp_name}")

    overrides.append(f"data.datamodule.dataset_path={args.dataset}")
    # overrides.append(f"datamodule.task={args.task}")

    # append all user-provided configuration overrides
    overrides += args.config

    initialize(config_path="../../../configurations/")
    conf = compose(config_name=config_name, overrides=overrides)

    # cannot call main() here because it's wrapped by hydra (...right? TODO: check)
    fix(conf)
    train(conf)


def test(cmd):
    import sys

    sys.argv = cmd.split(" ")
    print(cmd, end=" -> \n\t")
    main(parse_args())


if __name__ == "__main__":
    main(parse_args())
    # test("train.py sentence-pair data/glue/mrpc")
    # test("train.py token data/mrpc -m small -n mrpc-small")
    # test(
    #     "train.py token data/mrpc -m small "
    #     "-c training.pl_trainer.val_check_interval=1.0 data.pl_module.batch_size=16"
    # )
    # test("train.py sentence data/s")
