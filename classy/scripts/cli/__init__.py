from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("action")

    return parser.parse_args()


if __name__ == "__main__":
    pass
