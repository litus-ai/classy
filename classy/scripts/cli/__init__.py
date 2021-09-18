from argparse import ArgumentParser

import argcomplete


def get_commands():
    from classy.scripts.cli.train import get_parser as train_parser, main as train_main
    from classy.scripts.cli.predict import get_parser as predict_parser, main as predict_main
    from classy.scripts.cli.evaluate import get_parser as evaluate_parser, main as evaluate_main
    from classy.scripts.cli.serve import get_parser as serve_parser, main as serve_main
    from classy.scripts.cli.demo import get_parser as demo_parser, main as demo_main
    from classy.scripts.cli.describe import get_parser as describe_parser, main as describe_main

    return dict(
        train=dict(
            parser=train_parser,
            main=train_main,
        ),
        predict=dict(
            parser=predict_parser,
            main=predict_main,
        ),
        evaluate=dict(
            parser=evaluate_parser,
            main=evaluate_main,
        ),
        serve=dict(
            parser=serve_parser,
            main=serve_main,
        ),
        demo=dict(
            parser=demo_parser,
            main=demo_main,
        ),
        describe=dict(
            parser=describe_parser,
            main=describe_main
        ),
    )


def parse_args(commands: dict):

    parser = ArgumentParser()

    subcmds = parser.add_subparsers(dest="action", required=True)
    for action_name, action_data in commands.items():
        action_data["parser"](subcmds)

    argcomplete.autocomplete(parser, default_completer=None, always_complete_options="long")

    return parser.parse_args()


def main():
    commands = get_commands()
    args = parse_args(commands)

    commands[args.action]["main"](args)


if __name__ == "__main__":
    main()
