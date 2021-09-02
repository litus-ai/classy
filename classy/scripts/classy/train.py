from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("type", choices=("sequence", "token", "sentence-pair"))
    parser.add_argument("dataset", type=Path)
    parser.add_argument("-m", "--model-name", default="large")
    parser.add_argument("-n", "--name", "--exp-name", "--experiment-name", default=None)
    parser.add_argument("-d", "--device", default="gpu")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("-c", "--config", nargs="+", default=[])

    return parser.parse_args()


def validate_args(args):
    pass


def main():
    args = parse_args()
    validate_args(args)
    print(args)


def test(cmd):
    import sys

    sys.argv = cmd.split(" ")
    print(cmd, end=" -> \n\t")
    main()


if __name__ == "__main__":

    test("train.py sequence data/mrpc")
    test("train.py token data/mrpc -m small -n mrpc-small")
    test(
        "train.py token data/mrpc -m small "
        "-c training.pl_trainer.val_check_interval=1.0 data.pl_module.batch_size=16"
    )
    test("train.py sentence data/s")
