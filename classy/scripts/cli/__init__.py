from argparse import ArgumentParser


def get_commands():
    from classy.scripts.cli.train import get_parser as train_parser, main as train_main

    return dict(
        train=dict(
            parser=train_parser,
            main=train_main,
        ),
    )


def parse_args(commands: dict):

    parser = ArgumentParser()

    subcmds = parser.add_subparsers()
    for action_name, action_data in commands.items():
        action_data["parser"](subcmds)

    return parser.parse_args()


def main():
    import sys

    action = sys.argv[1]

    commands = get_commands()
    args = parse_args(commands)

    commands[action]["main"](args)


if __name__ == "__main__":
    main()
