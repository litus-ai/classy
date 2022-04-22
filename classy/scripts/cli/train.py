import copy
import json
import os
import shutil
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf

from classy.data.data_drivers import GENERATION, QA, SENTENCE_PAIR, SEQUENCE, TOKEN
from classy.scripts.cli.utils import get_device, maybe_find_directory
from classy.utils.help_cli import HELP_TASKS
from classy.utils.hydra_patch import ConfigBlame
from classy.utils.log import get_project_logger

logger = get_project_logger(__name__)


def populate_parser(parser: ArgumentParser):

    parser.add_argument(
        "task",
        choices=[SEQUENCE, SENTENCE_PAIR, TOKEN, QA, GENERATION],
        help=HELP_TASKS,
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="""
            Can be either a directory path containing a Training, a Validation and optionally a Test dataset, or a file
            path. In the latter case, classy will split the file in order to produce even a Validation anda Test set.
        """,
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="""
            Can be either a path to a profile (e.g. "/my-project/profiles/my-profile.yaml"), the name of a profile you
            created (i.e. the file name without the .yaml such as "my-profile") located in the "configurations/profiles"
            folder from your current path or a predefined profile. For a complete list of classy predefined profiles,
            please refer to the documentation.
        """,
    )
    parser.add_argument(
        "--transformer-model",
        type=str,
        default=None,
        help="""
            If you are using a transformer-based architecture, you can change the transformer model here using a valid
            name for the huggingface model-hub (e.g. roberta-base). For a complete list of available transformers please
            visit the model-hub official website at: "https://huggingface.co/models".
        """,
    )
    parser.add_argument(
        "-n",
        "--exp-name",
        "--experiment-name",
        dest="exp_name",
        required=True,
        help="""
            The name of the experiment. The checkpoints and the additional data for the runs will be stored under the
            "experiments/experiment-name" directory.
        """,
    )
    parser.add_argument(
        "-d",
        "--device",
        default=None,
        help="The device you will use for the training of your model. If not provided, classy will try to infer the desired behavior from the available environment.",
    )  # TODO: add validator?
    parser.add_argument(
        "-cn",
        "--config-name",
        default=None,
        help="The root of the hydra config files. (Probably you should not use this parameter.)",
    )
    parser.add_argument(
        "-cd",
        "--config-dir",
        default=None,
        help="""
            If you want to change the configuration directory you have to specify it here.
        """,
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="+",
        default=[],
        help="""
            Use this parameter to change anything in you configuration. To change the learning rate and the maximum
            number of steps you can do the following:
            "-c model.optim_conf.lr=0.0001 training.pl_traniner.max_steps=10_000".
            You can use --print parameter to find the parameters you want to modify.
        """,
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="The maximum number of epochs."
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="A checkpoint path from which you want to resume the training.",
    )
    parser.add_argument(
        "--wandb",
        nargs="?",
        const="anonymous",
        type=str,
        help="""
            If you want to log the training metrics on wandb, you can either only use "--wandb" and log the run as an
            anonymous one or you can use "--wandb project_name@experiment_name" and the run will be automatically logged
             into your account under the "project_name" project and with the "experiment_name" name.
        """,
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="No shuffling will be performed on the training data.",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="The training will use 16-bit precision."
    )
    parser.add_argument(
        "--vocabulary-dir",
        default=None,
        help="If you already computed the vocabulary, you can specify the directory here.",
    )
    parser.add_argument(
        "--big-dataset",
        action="store_true",
        help="""
            The training will follow some policies to handle large datasets,
            for more info please referer to the documentation.
        """,
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print all the training parameters following a tree structure.",
    )


def get_parser(subparser=None) -> ArgumentParser:

    parser_kwargs = dict(
        name="train",
        description="train a model with classy",
        help="Train a model with classy.",
    )
    parser = (subparser.add_parser if subparser is not None else ArgumentParser)(
        **parser_kwargs
    )

    populate_parser(parser)

    return parser


class ClassyBlame(ConfigBlame):
    def __init__(self, arg: str):  # noqa
        self.arg = arg

    def __str__(self):
        return f"[classy train [...] {self.arg}]"

    def __repr__(self):
        return self.__str__()


def parse_args():
    return get_parser().parse_args()


def _main_mock(cfg, blames: Optional[List] = None):

    dry_run_for_print = blames is not None
    blames = blames or []

    # import here to avoid importing torch before it's actually needed
    import hydra

    from classy.scripts.model.train import train
    from classy.utils.hydra import fix_paths

    if "supported_tasks" in cfg and cfg.task not in cfg.supported_tasks:
        logger.error(
            f"The profile you selected does not support the input task. "
            f"The following tasks are the ones supported: {', '.join(cfg.supported_tasks)}"
        )
        exit(1)

    if dry_run_for_print:
        from classy.utils.rich_config import print_config

        print_config(cfg, blames)
    else:
        # fix paths
        fix_paths(
            cfg,
            check_fn=lambda path: os.path.exists(
                hydra.utils.to_absolute_path(path[: path.rindex("/")])
            ),
            fix_fn=lambda path: hydra.utils.to_absolute_path(path),
        )
        train(cfg)


def _main_resume(model_dir: str):

    if not os.path.isdir(model_dir):
        logger.error(
            f"The previous run directory provided: '{model_dir}' does not exist."
        )
        exit(1)

    if not os.path.isfile(f"{model_dir}/checkpoints/last.ckpt"):
        logger.error(
            "The directory must contain the last checkpoint stored in the previous run (checkpoints/last.ckpt)."
        )
        exit(1)

    # import here to avoid importing torch before it's actually needed
    import sys

    import hydra
    from omegaconf import OmegaConf

    from classy.utils.lightning import load_training_conf_from_checkpoint

    model_dir = Path(model_dir)

    last_ckpt_path = model_dir / "checkpoints/last.ckpt"
    cfg = load_training_conf_from_checkpoint(
        str(last_ckpt_path), post_trainer_init=True
    )
    cfg.training.resume_from = str(last_ckpt_path)

    resuming_config_path = model_dir / ".hydra/resuming-config.yaml"
    OmegaConf.save(cfg, resuming_config_path)

    sys.argv = [
        "classy-train",
        "-cn",
        resuming_config_path.stem,
        "-cd",
        str(resuming_config_path.parent),
    ]
    hydra.main(config_path=None)(_main_mock)()


def classy_merge(
    base_cfg: DictConfig,
    updating_cfg: DictConfig,
    reference_folder: Optional[str] = None,
) -> List[str]:

    changes = []

    def is_interpolation(key):
        # OmegaConf.is_interpolation(base_cfg, key) requires key to be a direct children of base_cfg
        # this function patch this behavior so as to support any successor
        parent = base_cfg
        if "." in key:
            parent = OmegaConf.select(base_cfg, key[: key.rindex(".")])
            key = key[key.rindex(".") + 1 :]
        return OmegaConf.is_interpolation(parent, key)

    def rec(node, key):

        original_node = OmegaConf.select(base_cfg, key)

        # check if node is interpolation and, if it is, duplicate it
        if original_node is not None and is_interpolation(key):
            OmegaConf.update(
                base_cfg, key, copy.deepcopy(original_node), force_add=True
            )

        # check if node refers to a config group and, if it does, read it
        # and replace node
        if type(node) == str and reference_folder is not None:
            # check whether key is a primitive or if it refers to a config group
            for extension in ["yaml", "yml"]:
                config_group_path = (
                    Path(reference_folder)
                    / key.replace(".", "/")
                    / f"{node}.{extension}"
                )
                if config_group_path.exists():
                    # update node to be the config in the file
                    node = OmegaConf.load(config_group_path)
                    break

        # handle node
        if type(original_node) != type(node):
            # if types differ, overwrite
            # note that this is also handle the case when one is None and the other is not
            OmegaConf.update(base_cfg, key, node, merge=False, force_add=True)
            changes.append(key)
        elif OmegaConf.is_dict(node):
            if "_target_" in node:
                # overwrite dictionary
                OmegaConf.update(base_cfg, key, node, merge=False, force_add=True)
                changes.append(key)
            else:
                # recurse
                for k, v in node.items():
                    rec(v, f"{key}.{k}")
        elif OmegaConf.is_list(node):
            # append
            OmegaConf.update(
                base_cfg,
                key,
                OmegaConf.select(base_cfg, key) + node,
                merge=False,
                force_add=True,
            )
            changes.append(key)
        elif type(node) in [float, int, bool, str]:
            # overwrite
            OmegaConf.update(base_cfg, key, node, force_add=True)
            changes.append(key)
        else:
            raise ValueError(f"Unexpected type {type(node)}: {node}")

    for k, v in updating_cfg.items():
        rec(v, k)

    return changes


def apply_profiles_and_cli(
    config_name: str,
    config_dir: str,
    profile_path: Optional[str],
    cli: Dict[ClassyBlame, List[str]],
    output_config_name: str,
    is_profile_path: bool = False,
) -> List[Tuple[List[str], ClassyBlame]]:
    def parse_primitive(s: str):
        o = s
        if "." in s:
            try:
                o = float(s)
            except ValueError:
                o = s
        elif s.isdigit():
            try:
                o = int(s)
            except ValueError:
                o = s
        elif s.lower() in ["true", "false"]:
            return s.lower() == "true"
        else:
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return s
        return o

    # load initial configuration from folder
    with hydra.initialize_config_dir(config_dir=config_dir, job_name="train"):
        base_cfg = hydra.compose(config_name=config_name, return_hydra_config=True)
        blames = base_cfg.__dict__["_blame"]

    # load profile
    profile_cfg = (
        OmegaConf.load(profile_path)
        if profile_path is not None
        else OmegaConf.create({})
    )

    # apply cli edits on profile
    cli_changes = set()
    for blame, changes in cli.items():
        _changes = []
        for change in changes:
            _k, _v = change.split("=")
            _changes.extend(
                classy_merge(profile_cfg, OmegaConf.create({_k: parse_primitive(_v)}))
            )
        cli_changes.update(_changes)
        blames.append((_changes, blame))

    # apply edits over base_cfg
    profile_changes = classy_merge(base_cfg, profile_cfg, reference_folder=config_dir)

    # add profile blames (if profile was passed)
    if profile_path is not None:
        profile_blame_name = str(profile_path)
        if not is_profile_path:
            profile_blame_name = profile_blame_name.split("/")[-1]
            profile_blame_name = profile_blame_name[: profile_blame_name.rindex(".")]
        blames.append(
            (
                [change for change in profile_changes if change not in cli_changes],
                ClassyBlame(f"--profile {profile_blame_name}"),
            )
        )

    # save and save
    OmegaConf.save(base_cfg, f"{config_dir}/{output_config_name}.yaml")
    return blames


def handle_device(
    args, profile_path: Optional[Path], cli_overrides: Dict[ClassyBlame, List[str]]
):

    import torch.cuda

    # read profile
    profile_cfg = (
        OmegaConf.load(profile_path)
        if profile_path is not None
        else OmegaConf.create({})
    )

    # check that either user used cli params (-d 0, --fp16, ...) or profile or neither of the two
    profile_overrides_device = "device" in profile_cfg
    cli_specifies_device = args.device is not None
    assert int(profile_overrides_device) + int(cli_specifies_device) in [
        0,
        1,
    ], f"You are specifying your device in both profile ({profile_cfg.device}) and cli (-d {args.device}. This is not supported: either specify everything in profile or use only CLI"

    if profile_overrides_device:
        # here we can do nothing and return as profile logic will handle it
        return
    elif cli_specifies_device:
        # here we need to side-effect on cli_overrides
        device = get_device(args.device)
        if device != -1:
            if args.fp16:
                cli_overrides[ClassyBlame(f"-d {args.device} --fp16")] = [
                    f"training.pl_trainer.gpus=[{device}]",
                    "training.pl_trainer.precision=16",
                ]
            else:
                cli_overrides[ClassyBlame(f"-d {args.device}")] = [
                    f"training.pl_trainer.gpus=[{device}]",
                    "training.pl_trainer.precision=32",
                ]
        else:
            # user requested to train on cpu
            # we just need to check that fp16 has not been requested
            if args.fp16:
                logger.error("fp16 is only available when training on a GPU")
                exit(1)
    else:
        # auto-mode: infer what we should be doing from the environment
        # todo should this code be improved?
        if torch.cuda.is_available():
            # use first gpu
            cli_overrides[ClassyBlame(f"[-d {args.device}]")] = [
                "training.pl_trainer.accelerator=gpu",
                "training.pl_trainer.devices=1",
                "training.pl_trainer.auto_select_gpus=True",
            ]
        else:
            # use cpu
            pass


def main(args):

    import classy

    if args.resume_from is not None:
        _main_resume(args.resume_from)
        return

    with tempfile.TemporaryDirectory() as tmp_dir:

        # hydra config name and config dir
        config_name = args.config_name or args.task
        config_dir = args.config_dir or maybe_find_directory(
            [
                "configuration",
                "configurations",
                "config",
                "configs",
                "conf",
                "confs",
            ]
        )

        # copy config dir and installed classy configurations into tmp_dir
        classy_dir = str(Path(classy.__file__).parent.parent / "configurations")
        shutil.copytree(classy_dir, tmp_dir, dirs_exist_ok=True)
        if config_dir is not None:
            shutil.copytree(config_dir, tmp_dir, dirs_exist_ok=True)
        assert (
            Path(tmp_dir) / (config_name + ".yaml")
        ).exists(), f"No config name file {config_name} found in temporary config dir"

        # read user-provided configuration overrides from cli
        cli_overrides = {}

        # read args.config
        for override in args.config:
            if "~" in override:
                raise NotImplementedError('Usage of "~" is currently not supported')
            cli_overrides[ClassyBlame(f"-c {override}")] = [override]

        # add dataset path
        cli_overrides[ClassyBlame(f"{args.dataset}")] = [
            f"data.datamodule.dataset_path={args.dataset}"
        ]

        # turn off shuffling if requested
        if args.no_shuffle:
            cli_overrides[ClassyBlame("--no-shuffle")] = [
                "data.datamodule.shuffle_dataset=False"
            ]

        # set number of epochs
        if args.epochs:
            cli_overrides[ClassyBlame(f"--epochs {args.epochs}")] = [
                f"training.pl_trainer.max_epochs={args.epochs}"
            ]

        # setup wandb logging
        if args.wandb is not None:
            wandb_overrides = []
            wandb_overrides.append(f"logging.wandb.use_wandb=True")

            if args.wandb == "anonymous":
                wandb_overrides.append(f"logging.wandb.anonymous=allow")
                wandb_blame = ClassyBlame("--wandb anonymous")
            else:
                if "@" not in args.wandb:
                    print(
                        "If you specify a value for '--wandb' it must contain both the name of the "
                        "project and the name of the specific experiment in the following format: "
                        "'<project-name>@<experiment-name>'"
                    )
                    exit(1)

                project, experiment = args.wandb.split("@")
                wandb_overrides.append(f"logging.wandb.project_name={project}")
                wandb_overrides.append(f"logging.wandb.experiment_name={experiment}")
                wandb_blame = ClassyBlame(f"--wandb {args.wandb}")

            cli_overrides[wandb_blame] = wandb_overrides

        # change the underlying transformer model
        if args.transformer_model is not None:
            cli_overrides[
                ClassyBlame(f"--transformer-model {args.transformer_model}")
            ] = [f"transformer_model={args.transformer_model}"]

        # check if the user provided a pre-computed vocabulary
        if args.vocabulary_dir is not None:
            cli_overrides[ClassyBlame(f"--vocabulary-dir {args.vocabulary_dir}")] = [
                f"data.vocabulary_dir={args.vocabulary_dir}"
            ]

        # bid-dataset option
        if args.big_dataset:
            logger.info(
                "The --big-dataset option has been selected. "
                "Hence we will: 1) assume the training dataset is ALREADY SHUFFLED "
                "2) evaluate the model performance every 2 thousand steps"
                "3) If the dataset provided is a file path when splitting the whole dataset in train, validation and test"
                "we will partition with the following ratio: 0.90 / 0.05 / 0.05"
            )
            cli_overrides[ClassyBlame("--big-dataset")] = [
                "data.datamodule.shuffle_dataset=False",
                "training.pl_trainer.val_check_interval=2000",  # TODO: 2K steps seems quite arbitrary
                "data.datamodule.validation_split_size=0.05",
                "data.datamodule.test_split_size=0.05",
            ]

        # read profile
        profile, profile_path, is_profile_path = args.profile, None, False
        if profile is not None:
            if profile.endswith(".yaml") or profile.endswith(".yml"):
                logger.info(f"Passed profile {profile} was detected to be a path")
                profile_path = Path(profile)
                is_profile_path = True
                assert (
                    profile_path.exists()
                ), f"Passed profile {profile} does not seem to exist"
            else:
                for extension in ["yaml", "yml"]:
                    profile_path = (
                        Path(tmp_dir) / "profiles" / (profile + f".{extension}")
                    )
                    if profile_path.exists():
                        break
                assert (
                    profile_path.exists()
                ), f"Passed profile {profile} does not seem to exist"

        # handle device
        handle_device(args, profile_path=profile_path, cli_overrides=cli_overrides)

        # set experiment name
        cli_overrides[ClassyBlame(f"-n {args.exp_name}")] = [
            f"exp_name={args.exp_name}"
        ]

        # apply profile and cli overrides
        blames = apply_profiles_and_cli(
            config_name=config_name,
            config_dir=tmp_dir,
            profile_path=profile_path,
            is_profile_path=is_profile_path,
            cli=cli_overrides,
            output_config_name="run",
        )

        # setup cmd
        cmd = ["classy-train", "-cn", "run", "-cd", tmp_dir]

        # we wrap this under a try-catch block because streamlit is an optional dependency
        try:

            # we import streamlit so that the stderr handler is added to the root logger here and we can remove it
            # it was imported in task_ui.py and was double-logging stuff...
            # this is the best workaround at this time, but we should investigate and / or (re-)open an issue
            # https://github.com/streamlit/streamlit/issues/1248
            import logging

            import streamlit

            with open("/dev/null", "w") as f:
                # we do this here so that streamlit's import is not unused and is not removed by linters & co
                print(streamlit.__version__, file=f)

            # at this point, streamlit's is the only handler added, so we can safely reset the handlers
            logging.getLogger().handlers = []

        except ImportError:
            # nothing to do then
            pass

        import sys

        import hydra

        # we are basically mocking the normal python script invocation by setting the argv to those we want
        # unfortunately there is no better way to do this at this moment in time :(
        sys.argv = cmd
        hydra.main(config_path=None)(
            lambda cfg: _main_mock(cfg, blames=blames if args.print else None)
        )()


if __name__ == "__main__":
    main(parse_args())
