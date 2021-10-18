import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional

from omegaconf import OmegaConf

from classy.scripts.cli.utils import get_device, maybe_find_directory
from classy.utils.hydra_patch import ConfigBlame
from classy.utils.log import get_project_logger

logger = get_project_logger(__name__)


def populate_parser(parser: ArgumentParser):

    # TODO: add help?
    parser.add_argument("task", choices=("sequence", "token", "sentence-pair", "qa", "generation"))
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--transformer-model", type=str, default=None)
    parser.add_argument("-n", "--exp-name", "--experiment-name", dest="exp_name", required=True)
    parser.add_argument("-d", "--device", default="gpu")  # TODO: add validator?
    parser.add_argument("-cn", "--config-name", default=None)
    parser.add_argument("-cd", "--config-dir", default=None)
    parser.add_argument("-c", "--config", nargs="+", default=[])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume-from", type=str)
    parser.add_argument("--wandb", nargs="?", const="anonymous", type=str)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--vocabulary-dir", default=None)
    parser.add_argument("--big-dataset", action="store_true")
    parser.add_argument("--print", action="store_true")


def get_parser(subparser=None) -> ArgumentParser:

    parser_kwargs = dict(name="train", description="train a model with classy", help="TODO")
    parser = (subparser.add_parser if subparser is not None else ArgumentParser)(**parser_kwargs)

    populate_parser(parser)

    return parser


class ClassyBlame(ConfigBlame):
    def __init__(self, arg: str):  # noqa
        self.arg = arg

    def __str__(self):
        return f"[classy train [...] {self.arg}]"


def parse_args():
    return get_parser().parse_args()


def _main_print_config(blames):
    from classy.utils.rich_config import print_config

    def _print_config(cfg):
        print_config(cfg, blames)

    return _print_config


def _main_mock(
    cfg, profile_name: Optional[str] = None, cli_overrides: Optional[List[str]] = None, blames: Optional[List] = None
):

    dry_run_for_print, blames = blames is not None, blames or []

    # import here to avoid importing torch before it's actually needed
    import hydra
    from classy.scripts.model.train import fix_paths, train

    # if cli_overrides is not None, build mapping cli_override_key -> result
    cli_override2result = {}
    if cli_overrides is not None:
        cli_override2result = {}
        for k in cli_overrides:
            cli_override2result[k] = OmegaConf.select(cfg, k)

    # apply profile overrides
    if "profiles" in cfg:

        subtrees_impacted_by_profile_change = set()

        def override_subtree(node, prefix: str):
            if OmegaConf.is_list(node):
                # if profiles override a list, the original list should be completely overwritten
                OmegaConf.update(cfg, prefix, node, merge=False, force_add=True)
                blames.append(([prefix], ClassyBlame(f"--profile {profile_name}")))
                subtrees_impacted_by_profile_change.add(prefix)
            elif OmegaConf.is_dict(node):
                # if profiles override a dict, the original dict should be discarded if _target_ is changed, and updated otherwise
                target_node = OmegaConf.select(cfg, prefix)
                if target_node is None:
                    OmegaConf.update(cfg, prefix, node, force_add=True)
                    blames.append(([prefix], ClassyBlame(f"--profile {profile_name}")))
                    subtrees_impacted_by_profile_change.add(prefix)
                else:
                    if "_target_" in node:
                        OmegaConf.update(cfg, prefix, node, merge=False, force_add=True)
                        blames.append(([prefix], ClassyBlame(f"--profile {profile_name}")))
                        subtrees_impacted_by_profile_change.add(prefix)
                    else:
                        for k, v in node.items():
                            override_subtree(v, prefix=f"{prefix}.{k}")
            elif type(node) in [str, float, int, bool] or node is None:
                if prefix not in cli_override2result:
                    OmegaConf.update(cfg, prefix, node, force_add=True)
                    blames.append(([prefix], ClassyBlame(f"--profile {profile_name}")))
                    subtrees_impacted_by_profile_change.add(prefix)
            else:
                raise ValueError(f"Unexpected type {type(node)}: {node}")

        profile = cfg.profiles
        del cfg.profiles
        for k, n in profile.items():
            override_subtree(n, prefix=k)

        # re-apply overrides
        def is_subtree(st1, st2) -> bool:
            st1, st2 = st1.split("."), st2.split(".")
            return len(st1) > len(st2) and all(_st1 == _st2 for _st1, _st2 in zip(st1, st2))

        for k, v in cli_override2result.items():
            # re-apply v
            # note that this delete changes applied by profile if profile changed a subgraph (e.g x.y) later changed by a cli override (e.g. x)
            # this is what the following warning checks
            for _st in subtrees_impacted_by_profile_change:
                assert not is_subtree(_st, k), f"{_st}, changed by profile, is a subtree of {k}, changed by CLI"
            OmegaConf.update(cfg, k, v, merge=False)

    # fix paths
    fix_paths(
        cfg,
        check_fn=lambda path: os.path.exists(hydra.utils.to_absolute_path(path[: path.rindex("/")])),
        fix_fn=lambda path: hydra.utils.to_absolute_path(path),
    )

    if "supported_tasks" in cfg and cfg.task not in cfg.supported_tasks:
        logger.error(
            f"The profile you selected does not support the input task. "
            f"The following tasks are the ones supported: {', '.join(cfg.supported_tasks)}"
        )
        exit(1)

    if dry_run_for_print:
        _main_print_config(blames)(cfg)
    else:
        train(cfg)


def _main_resume(model_dir: str):

    if not os.path.isdir(model_dir):
        logger.error(f"The previous run directory provided: '{model_dir}' does not exist.")
        exit(1)

    if not os.path.isfile(f"{model_dir}/checkpoints/last.ckpt"):
        logger.error(
            "The directory must contain the last checkpoint stored in the previous run (checkpoints/last.ckpt)."
        )
        exit(1)

    # import here to avoid importing torch before it's actually needed
    import hydra
    import sys
    from omegaconf import OmegaConf

    from classy.utils.lightning import load_training_conf_from_checkpoint

    model_dir = Path(model_dir)

    last_ckpt_path = model_dir / "checkpoints/last.ckpt"
    cfg = load_training_conf_from_checkpoint(str(last_ckpt_path), post_trainer_init=True)
    cfg.training.resume_from = str(last_ckpt_path)

    resuming_config_path = model_dir / ".hydra/resuming-config.yaml"
    OmegaConf.save(cfg, resuming_config_path)

    sys.argv = ["classy-train", "-cn", resuming_config_path.stem, "-cd", str(resuming_config_path.parent)]
    hydra.main(config_path=None)(_main_mock)()


def main(args):
    if args.resume_from is not None:
        _main_resume(args.resume_from)
        return

    cmd = ["classy-train", "-cn", args.config_name or args.task]
    blames = []

    conf_dir = args.config_dir or maybe_find_directory(
        [
            "configuration",
            "configurations",
            "config",
            "configs",
            "conf",
            "confs",
        ]
    )

    if conf_dir is not None:
        cmd += ["-cd", conf_dir]

    # override all the fields modified by the profile
    if args.profile is not None:
        cmd.append(f"+profiles={args.profile}")

    # choose device
    device = get_device(args.device)
    if device >= 0:
        if args.fp16:
            cmd.append("device=cuda_amp")
        else:
            cmd.append(f"device=cuda")
        cmd.append(f"device.gpus=[{device}]")
    else:
        if args.fp16:
            logger.error("fp16 is only available when training on a GPU")
            return
        cmd.append(f"device=cpu")

    cmd.append(f"exp_name={args.exp_name}")

    # add dataset path
    cmd.append(f"data.datamodule.dataset_path={args.dataset}")

    # turn off shuffling if requested
    if args.no_shuffle:
        cmd.append("data.datamodule.shuffle_dataset=False")
        blames.append((["data.datamodule.shuffle_dataset"], ClassyBlame("--no-shuffle")))

    if args.epochs:
        cmd.append(f"training.pl_trainer.max_epochs={args.epochs}")

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
            configs.extend(("logging.wandb.project_name", "logging.wandb.experiment_name"))
            to_blame = ClassyBlame(f"--wandb {args.wandb}")

        blames.append((configs, to_blame))

    # change the underlying transformer model
    if args.transformer_model is not None:
        cmd.append(f"transformer_model={args.transformer_model}")
        blames.append((["transformer_model"], ClassyBlame(f"--transformer-model {args.transformer_model}")))

    # precomputed vocabulary from the user
    if args.vocabulary_dir is not None:
        cmd.append(f"data.vocabulary_dir={args.vocabulary_dir}")
        blames.append((["data.vocabulary_dir"], ClassyBlame(f"--vocabulary-dir {args.vocabulary_dir}")))

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
        cmd.append("training.pl_trainer.val_check_interval=2000")  # TODO: 2K steps seems quite arbitrary
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

    # we import streamlit so that the stderr handler is added to the root logger here and we can remove it
    # it was imported in task_ui.py and was double-logging stuff...
    # this is the best workaround at this time, but we should investigate and / or (re-)open an issue
    # https://github.com/streamlit/streamlit/issues/1248
    import logging

    # at this point, streamlit's is the only handler added, so we can safely reset the handlers
    logging.getLogger().handlers = []

    import hydra
    import sys

    # compute cli overrides and check that only primitive types have been used
    cli_overrides = [c.split("=")[0] for c in cmd[6:]]

    # we are basically mocking the normal python script invocation by setting the argv to those we want
    # unfortunately there is no better way to do this at this moment in time :(
    sys.argv = cmd
    hydra.main(config_path=None)(
        lambda cfg: _main_mock(
            cfg, profile_name=args.profile, cli_overrides=cli_overrides, blames=blames if args.print else None
        )
    )()


if __name__ == "__main__":
    main(parse_args())
