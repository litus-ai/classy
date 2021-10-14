import os
from argparse import ArgumentParser
from pathlib import Path

from omegaconf import OmegaConf, open_dict

from classy.scripts.cli.utils import get_device
from classy.utils.log import get_project_logger

logger = get_project_logger(__name__)


def populate_parser(parser: ArgumentParser):

    # TODO: add help?
    parser.add_argument("task", choices=("sequence", "token", "sentence-pair", "qa", "generation"))
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--transformer-model", type=str, default=None)
    parser.add_argument("-n", "--exp-name", "--experiment-name", dest="exp_name", default=None)
    parser.add_argument("-d", "--device", default="gpu")  # TODO: add validator?
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("-c", "--config", nargs="+", default=[])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume-from", type=str)
    parser.add_argument("--wandb", nargs="?", const="anonymous", type=str)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--vocabulary-dir", default=None)
    parser.add_argument("--big-dataset", action="store_true")


def get_parser(subparser=None) -> ArgumentParser:

    parser_kwargs = dict(name="train", description="train a model with classy", help="TODO")
    parser = (subparser.add_parser if subparser is not None else ArgumentParser)(**parser_kwargs)

    populate_parser(parser)

    return parser


def parse_args():
    return get_parser().parse_args()


def _main_mock(cfg):
    # import here to avoid importing torch before it's actually needed
    import hydra
    import omegaconf
    from classy.scripts.model.train import fix_paths, train

    # apply profile

    def override_subtree(node, prefix: str):
        if OmegaConf.is_list(node):
            # if profiles override a list, the original list should be completely overwritten
            OmegaConf.update(cfg, prefix, node, merge=False, force_add=True)
        elif OmegaConf.is_dict(node):
            # if profiles override a dict, the original dict should be discarded if _target_ is changed, and updated otherwise
            target_node = OmegaConf.select(cfg, prefix)
            if target_node is None:
                OmegaConf.update(cfg, prefix, node, force_add=True)
            else:
                if '_target_' in node:
                    OmegaConf.update(cfg, prefix, node, merge=False, force_add=True)
                else:
                    for k, v in node.items():
                        override_subtree(v, prefix=f"{prefix}.{k}")
        elif type(node) in [str, float, int, bool] or node is None:
            OmegaConf.update(cfg, prefix, node, force_add=True)
        else:
            raise ValueError(f"Unexpected type {type(node)}: {node}")

    profile = cfg.profiles
    del cfg.profiles
    for k, n in profile.items():
        override_subtree(n, prefix=k)

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
    import hydra
    import sys

    if args.resume_from is not None:
        _main_resume(args.resume_from)
        return

    if args.root is not None:
        config_name = args.root
    else:
        config_name = args.task

    cmd = ["classy-train", "-cn", config_name, "-cd", str(Path.cwd() / "configurations")]

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

    # create default experiment name if not provided
    exp_name = args.exp_name or f"{args.task}-{args.model_name}"
    cmd.append(f"exp_name={exp_name}")

    # add dataset path
    cmd.append(f"data.datamodule.dataset_path={args.dataset}")

    # turn off shuffling if requested
    if args.no_shuffle:
        cmd.append("data.datamodule.shuffle_dataset=False")

    if args.epochs:
        cmd.append(f"+training.pl_trainer.max_epochs={args.epochs}")

    # wandb logging
    if args.wandb is not None:
        cmd.append(f"logging.wandb.use_wandb=True")
        if args.wandb == "anonymous":
            cmd.append(f"logging.wandb.anonymous=allow")
        else:
            assert "@" in args.wandb, (
                "If you specify a value for '--wandb' it must contain both the name of the "
                "project and the name of the specific experiment in the following format: "
                "'<project-name>@<experiment-name>'"
            )

            project, experiment = args.wandb.split("@")
            cmd.append(f"logging.wandb.project_name={project}")
            cmd.append(f"logging.wandb.experiment_name={experiment}")

    # change the underlying transformer model
    if args.transformer_model is not None:
        cmd.append(f"transformer_model={args.transformer_model}")

    # precomputed vocabulary from the user
    if args.vocabulary_dir is not None:
        cmd.append(f"+data.vocabulary_dir={args.vocabulary_dir}")

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
