from argparse import ArgumentParser
from pathlib import Path

from classy.scripts.cli.utils import get_device


def populate_parser(parser: ArgumentParser):

    # TODO: add help?
    parser.add_argument("task", choices=("sequence", "token", "sentence-pair"))
    parser.add_argument("dataset", type=Path)
    parser.add_argument("-m", "--model-name", default="bert")
    parser.add_argument("-n", "--exp-name", "--experiment-name", dest="exp_name", default=None)
    parser.add_argument("-d", "--device", default="gpu")  # TODO: add validator?
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("-c", "--config", nargs="+", default=[])


def get_parser(subparser=None) -> ArgumentParser:
    # subparser: Optional[argparse._SubParsersAction]

    parser_kwargs = dict(name="train", description="train a model with classy", help="TODO")
    parser = (subparser.add_parser if subparser is not None else ArgumentParser)(**parser_kwargs)

    populate_parser(parser)

    return parser


def parse_args():
    return get_parser().parse_args()


def _main_mock(cfg):
    # import here to avoid importing torch before it's actually needed
    from classy.scripts.model.train import fix, train

    fix(cfg)
    train(cfg)


def main(args):
    import hydra
    import sys

    if args.root is not None:
        config_name = args.root
    else:
        config_name = f"{args.task}-{args.model_name}"

    cmd = ["classy-train", "-cn", config_name, "-cd", str(Path.cwd() / "configurations")]

    # choose device
    device = get_device(args.device)
    if device >= 0:
        cmd.append(f"device=cuda")
        cmd.append(f"device.gpus=[{device}]")
    else:
        cmd.append(f"device=cpu")

    # create default experiment name if not provided
    exp_name = args.exp_name or f"{args.task}-{args.model_name}"
    cmd.append(f"exp_name={exp_name}")

    # add dataset path
    cmd.append(f"data.datamodule.dataset_path={args.dataset}")

    # append all user-provided configuration overrides
    cmd += args.config

    # we are basically mocking the normal python script invocation by setting the argv to those we want
    # unfortunately there is no better way to do this at this moment in time :(
    sys.argv = cmd
    hydra.main(config_path=None)(_main_mock)()


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
