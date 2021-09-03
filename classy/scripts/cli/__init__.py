from argparse import ArgumentParser


def get_commands():
    from classy.scripts.cli.train import get_parser as train_parser, main as train_main
    from classy.scripts.cli.predict import get_parser as predict_parser, main as predict_main

    return dict(
        train=dict(
            parser=train_parser,
            main=train_main,
        ),
        predict=dict(
            parser=predict_parser,
            main=predict_main,
        ),
    )


def parse_args(commands: dict):

    parser = ArgumentParser()

    subcmds = parser.add_subparsers(dest="action")
    for action_name, action_data in commands.items():
        action_data["parser"](subcmds)

    return parser.parse_args()


def main():
    commands = get_commands()
    args = parse_args(commands)

    commands[args.action]["main"](args)


if __name__ == "__main__":
    main()
