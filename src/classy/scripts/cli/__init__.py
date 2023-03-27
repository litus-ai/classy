from argparse import ArgumentParser
from pathlib import Path

import argcomplete

from ...utils.log import get_project_logger
from .utils import import_module_and_submodules, maybe_find_directory

logger = get_project_logger(__name__)


def get_commands():
    from .demo import get_parser as demo_parser
    from .demo import main as demo_main
    from .describe import get_parser as describe_parser
    from .describe import main as describe_main
    from .download import get_parser as download_parser
    from .download import main as download_main
    from .evaluate import get_parser as evaluate_parser
    from .evaluate import main as evaluate_main
    from .export import get_parser as export_parser
    from .export import main as export_main
    from .import_ import get_parser as import_parser
    from .import_ import main as import_main
    from .predict import get_parser as predict_parser
    from .predict import main as predict_main
    from .serve import get_parser as serve_parser
    from .serve import main as serve_main
    from .train import get_parser as train_parser
    from .train import main as train_main
    from .upload import get_parser as upload_parser
    from .upload import main as upload_main

    commands = dict(
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
            main=describe_main,
        ),
        download=dict(
            parser=download_parser,
            main=download_main,
        ),
        upload=dict(
            parser=upload_parser,
            main=upload_main,
        ),
        export=dict(parser=export_parser, main=export_main),
    )

    commands["import"] = dict(parser=import_parser, main=import_main)

    return commands


def parse_args(commands: dict):
    parser = ArgumentParser()

    grp = parser.add_mutually_exclusive_group()

    grp.add_argument(
        "--install-autocomplete",
        action="store_true",
        help="Installs classy's autocomplete (currently works with bash and zsh only)",
    )

    subcmds = parser.add_subparsers(dest="action", required=False)
    for action_name, action_data in commands.items():
        cmd_parser = action_data["parser"](subcmds)
        cmd_parser.add_argument("-pd", "--package-dir", default=None)

    argcomplete.autocomplete(
        parser, default_completer=None, always_complete_options="long"
    )

    return parser.parse_args()


def install_autocomplete():
    import os

    prefix = os.getenv("CONDA_PREFIX")
    if prefix is None:
        print(
            "CONDA_PREFIX unset. Are you sure you are executing this within a conda environment?"
        )
        return

    if "envs" not in prefix:
        print("CONDA_PREFIX does not appear to be an environment of conda.")
        print(
            "Are you sure you are executing this within a conda environment (not the base env)?"
        )
        print(f"   CONDA_PREFIX={prefix}")
        return

    path = Path(prefix)
    script_path = path / "etc/conda/activate.d/classy-complete.sh"
    if script_path.exists():
        print("Autocomplete already installed! Exiting...")
        return

    script_path.parent.mkdir(parents=True, exist_ok=True)

    # TODO: improve to make it work with other shells!
    # currently works with bash and zsh (if bashcompinit is enabled!)
    # see https://github.com/kislyuk/argcomplete#activating-global-completion for more info
    with script_path.open("w") as f:
        s = 'eval "$(register-python-argcomplete classy)"'
        print(s, file=f)

    print("Autocompletion installed, enjoy classy! :)")


def main():
    commands = get_commands()
    args = parse_args(commands)

    if args.install_autocomplete:
        install_autocomplete()
        return

    if args.action is None:
        print(
            "No action has been provided. Execute `classy --install-autocomplete` "
            "to install classy's shell completion or `classy -h` for help"
        )
        exit(1)

    # import classy components
    import_module_and_submodules("classy")

    # import user-defined component
    to_import = args.package_dir or maybe_find_directory(("src", "source"))
    if to_import is not None:
        to_import_path = Path(to_import)
        if "src/classy" not in [str(sf) for sf in to_import_path.iterdir()]:
            import_module_and_submodules(to_import)
        else:
            # we infer that we are developing on classy itself and avoid re-importing everything
            # (which will make things like omegaconf resolver registration crash, as they are already registered)
            logger.warning(
                f'Found "classy" subfolder in source folder "{to_import}". '
                f"This should only happen when developing on the library itself. "
                f"If this is not the case, please rename the module. "
                f"Otherwise, classy may (will, most likely) not function as expected."
            )

    # run command
    commands[args.action]["main"](args)


if __name__ == "__main__":
    main()
