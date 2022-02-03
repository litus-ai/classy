import os
import shutil
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf

import classy
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
        default="gpu",
        help="The device you will use for the training of your model.",
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


def apply_profile_on_dir(
    profile: DictConfig, profile_name: str, config_name: str, config_dir: str
):

    blames = []

    def recurse_and_fix(
        prefix,
        profile_node,
        cfg,
        blame_prefix,
        path_to_target_config,
        defaults,
        potential_defaults,
    ):

        if OmegaConf.is_dict(profile_node):
            # if profile overrides a dict, the original dict should be:
            target_node = OmegaConf.select(cfg, prefix)
            if target_node is None:
                # inserted if the dict was not present
                OmegaConf.update(cfg, prefix, profile_node, force_add=True)
                blames.append(
                    (
                        [(blame_prefix + "." + prefix).lstrip(".")],
                        ClassyBlame(f"--profile {profile_name}"),
                    )
                )
            else:
                if "_target_" in profile_node:
                    # discarded if _target_ is changed
                    if prefix == "":
                        cfg = profile_node
                        assert potential_defaults is not None
                        for k in potential_defaults:
                            if k in cfg:
                                assert type(cfg[k]) == str
                                defaults[k] = cfg[k]
                                cfg.pop(k)
                    else:
                        OmegaConf.update(
                            cfg, prefix, profile_node, merge=False, force_add=True
                        )
                    blames.append(
                        (
                            [(blame_prefix + "." + prefix).lstrip(".")],
                            ClassyBlame(f"--profile {profile_name}"),
                        )
                    )
                else:
                    # merged and updated recursively if it was present
                    for k, v in profile_node.items():
                        if potential_defaults is not None and k in potential_defaults:
                            # potential defaults is not None only for direct children of a root (where the defaults logic should be applied)
                            # if (k, v) refers to a new default
                            if type(v) == str:
                                # this should be added
                                defaults[k] = v
                            else:
                                # launch fixing logic on child file
                                child_file = (
                                    path_to_target_config.parent
                                    / k
                                    / (defaults[k] + ".yaml")
                                )
                                assert (
                                    child_file.exists()
                                ), f"{child_file} not found in config dir"
                                apply_recursively(
                                    v, child_file, (prefix + "." + k).lstrip(".")
                                )
                        else:
                            # otherwise, standard recursion
                            recurse_and_fix(
                                k,
                                v,
                                target_node,
                                (blame_prefix + "." + prefix).lstrip("."),
                                path_to_target_config=None,
                                defaults=None,
                                potential_defaults=None,
                            )
        elif OmegaConf.is_list(profile_node):
            # if profile overrides a list, the original list should be completely overwritten
            if prefix == "":
                cfg = profile_node
            else:
                OmegaConf.update(cfg, prefix, profile_node, merge=False, force_add=True)
            blames.append(
                (
                    [(blame_prefix + "." + prefix).lstrip(".")],
                    ClassyBlame(f"--profile {profile_name}"),
                )
            )
        elif type(profile_node) in [str, float, int, bool] or profile_node is None:
            OmegaConf.update(cfg, prefix, profile_node, force_add=True)
            blames.append(
                (
                    [(blame_prefix + "." + prefix).lstrip(".")],
                    ClassyBlame(f"--profile {profile_name}"),
                )
            )
        else:
            raise ValueError(f"Unexpected type {type(profile_node)}: {profile_node}")

        return cfg

    def apply_recursively(profile_node, path_to_target_config: Path, prefix: str):

        # load conf
        cfg = OmegaConf.load(path_to_target_config)

        # compute potential defaults dict (folders present)
        potential_defaults = set(
            [
                d.name
                for d in path_to_target_config.parent.iterdir()
                if d.is_dir() and d.name != "__pycache__"
            ]
        )

        # extract defaults dict
        is_self_first = None
        defaults = {}

        if "defaults" in cfg:
            for i, d in enumerate(cfg.defaults):
                if d == "_self_":
                    is_self_first = i == 0
                    continue
                for k, v in d.items():
                    assert (
                        k not in defaults
                    ), f"Key {k} already present in defaults list. Check your defaults list"
                    defaults[k] = v

        # check all defaults are in potential defaults
        assert all(d in potential_defaults for d in defaults)

        # apply profile
        cfg = recurse_and_fix(
            "",
            profile_node,
            cfg,
            blame_prefix=prefix,
            path_to_target_config=path_to_target_config,
            defaults=defaults,
            potential_defaults=potential_defaults,
        )

        # update defaults
        if len(defaults) > 0:
            cfg.defaults = [{k: v} for k, v in defaults.items()]
            if is_self_first is not None:
                if is_self_first:
                    cfg.defaults = ["_self_"] + cfg.defaults
                else:
                    cfg.defaults = cfg.defaults + ["_self_"]

        # update config
        OmegaConf.save(cfg, path_to_target_config)

    apply_recursively(profile, Path(config_dir) / (config_name + ".yaml"), prefix="")

    return blames


def main(args):

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

        # set blames list
        blames = []

        # copy config dir and installed classy configurations into tmp_dir
        classy_dir = str(Path(classy.__file__).parent.parent / "configurations")
        shutil.copytree(classy_dir, tmp_dir, dirs_exist_ok=True)
        if config_dir is not None:
            shutil.copytree(config_dir, tmp_dir, dirs_exist_ok=True)
        assert (
            Path(tmp_dir) / (config_name + ".yaml")
        ).exists(), f"No config name file {config_name} found in temporary config dir"

        # apply profile on config dir
        if args.profile is not None:
            if args.profile.endswith(".yaml") or args.profile.endswith(".yml"):
                logger.info(f"Passed profile {args.profile} was detected to be a path")
                profile_path = Path(args.profile)
            else:
                profile_path = Path(tmp_dir) / "profiles" / (args.profile + ".yaml")
            assert profile_path.exists(), f"No profile found at {profile_path}"
            blames += apply_profile_on_dir(
                OmegaConf.load(profile_path), args.profile, config_name, tmp_dir
            )

        cmd = ["classy-train", "-cn", args.config_name or args.task, "-cd", tmp_dir]

        # choose device
        device = get_device(args.device)
        if device != -1:
            if args.fp16:
                cmd.append("device=cuda_amp")
            else:
                cmd.append(f"device=cuda")
            if not isinstance(device, int) or device > 0:
                cmd.append(f"device.gpus=[{device}]")
        else:
            if args.fp16:
                logger.error("fp16 is only available when training on a GPU")
                return
            cmd.append(f"device=cpu")
        blames.append((["device"], ClassyBlame(f"-d {args.device}")))

        cmd.append(f"exp_name={args.exp_name}")
        blames.append((["exp_name"], ClassyBlame(f"-n {args.exp_name}")))

        # add dataset path
        cmd.append(f"data.datamodule.dataset_path={args.dataset}")
        blames.append(
            (["data.datamodule.dataset_path"], ClassyBlame(f"{args.dataset}"))
        )

        # turn off shuffling if requested
        if args.no_shuffle:
            cmd.append("data.datamodule.shuffle_dataset=False")
            blames.append(
                (["data.datamodule.shuffle_dataset"], ClassyBlame("--no-shuffle"))
            )

        if args.epochs:
            cmd.append(f"++training.pl_trainer.max_epochs={args.epochs}")
            blames.append(
                (
                    ["training.pl_trainer.max_epochs"],
                    ClassyBlame(f"--epochs {args.epochs}"),
                )
            )

        # wandb logging
        if args.wandb is not None:
            cmd.append(f"logging.wandb.use_wandb=True")
            configs = ["logging.wandb.use_wandb"]

            if args.wandb == "anonymous":
                cmd.append(f"logging.wandb.anonymous=allow")
                configs.append("logging.wandb.anonymous")
                to_blame = ClassyBlame("--wandb anonymous")
            else:
                if "@" not in args.wandb:
                    print(
                        "If you specify a value for '--wandb' it must contain both the name of the "
                        "project and the name of the specific experiment in the following format: "
                        "'<project-name>@<experiment-name>'"
                    )
                    exit(1)

                project, experiment = args.wandb.split("@")
                cmd.append(f"logging.wandb.project_name={project}")
                cmd.append(f"logging.wandb.experiment_name={experiment}")
                configs.extend(
                    ("logging.wandb.project_name", "logging.wandb.experiment_name")
                )
                to_blame = ClassyBlame(f"--wandb {args.wandb}")

            blames.append((configs, to_blame))

        # change the underlying transformer model
        if args.transformer_model is not None:
            cmd.append(f"transformer_model={args.transformer_model}")
            blames.append(
                (
                    ["transformer_model"],
                    ClassyBlame(f"--transformer-model {args.transformer_model}"),
                )
            )

        # precomputed vocabulary from the user
        if args.vocabulary_dir is not None:
            cmd.append(f"data.vocabulary_dir={args.vocabulary_dir}")
            blames.append(
                (
                    ["data.vocabulary_dir"],
                    ClassyBlame(f"--vocabulary-dir {args.vocabulary_dir}"),
                )
            )

        # bid-dataset option
        if args.big_dataset:
            logger.info(
                "The user selected the --big-dataset option. "
                "Hence we will: 1) assume the training dataset is ALREADY SHUFFLED "
                "2) evaluate the model performance every 2 thousand steps"
                "3) If the dataset provided is a file path when splitting the whole dataset in train, validation and test"
                "we will partition with the following ratio: 0.90 / 0.05 / 0.05"
            )
            cmd.append("data.datamodule.shuffle_dataset=False")
            cmd.append(
                "training.pl_trainer.val_check_interval=2000"
            )  # TODO: 2K steps seems quite arbitrary
            cmd.append("data.datamodule.validation_split_size=0.05")
            cmd.append("data.datamodule.test_split_size=0.05")
            blames.append(
                (
                    [
                        "data.datamodule.shuffle_dataset",
                        "training.pl_trainer.val_check_interval",
                        "data.datamodule.validation_split_size",
                        "data.datamodule.test_split_size",
                    ],
                    ClassyBlame("--big-dataset"),
                )
            )

        # append all user-provided configuration overrides
        cmd += args.config
        for override in args.config:
            key, value = override.split("=")
            key = key.lstrip("+~")
            blames.append(([key], ClassyBlame(f"-c {override}")))

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
