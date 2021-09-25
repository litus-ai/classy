from pathlib import Path
from typing import Dict

import omegaconf

import hydra
from omegaconf import OmegaConf, DictConfig

from classy.pl_modules.base import ClassyPLModule
from classy.utils.hydra import fix_paths
from classy.utils.vocabulary import Vocabulary

import logging

logger = logging.getLogger(__name__)


def load_training_conf_from_checkpoint(checkpoint_path: str, post_trainer_init: bool = False) -> DictConfig:
    # find hydra config path
    experiment_folder = Path(checkpoint_path).parent.parent
    # load hydra config
    conf_file = "config_post_trainer_init.yaml" if post_trainer_init else "config.yaml"
    conf = OmegaConf.load(f"{experiment_folder}/.hydra/{conf_file}".lstrip("/"))
    # fix paths
    fix_paths(
        conf,
        check_fn=lambda path: experiment_folder.joinpath(path).exists(),
        fix_fn=lambda path: str(experiment_folder.joinpath(path)),
    )
    # return
    return conf


def load_classy_module_from_checkpoint(checkpoint_path: str) -> ClassyPLModule:
    """
    Load a PL module from a checkpoint path only. Infer the model to load from the dumped hydra conf

    Args:
        checkpoint_path (str):

    Returns:
        pl.LightningModule

    """

    conf = load_training_conf_from_checkpoint(checkpoint_path)

    # check if the model requires a vocab
    train_dataset_class = conf["data"]["datamodule"]["train_dataset"]["_target_"]
    if not train_dataset_class.split(".")[-1][0].isupper():  # if it is not upper then it is a class method
        train_dataset_class = ".".join(train_dataset_class.split(".")[:-1])

    requires_vocab = hydra.utils.instantiate({"_target_": f"{train_dataset_class}.requires_vocab"})

    # extract and build vocabulary
    vocabulary_path = Path(checkpoint_path).parent.parent / "vocabulary"

    assert (not requires_vocab) or vocabulary_path.exists(), f"No vocabulary found at path {vocabulary_path}"

    vocabulary = None
    if vocabulary_path.exists():
        vocabulary = Vocabulary.from_folder(vocabulary_path)

    # instantiate and return
    instantiate_input = dict(checkpoint_path=checkpoint_path)
    if vocabulary is not None:
        instantiate_input["vocabulary"] = vocabulary

    return hydra.utils.instantiate(
        {"_target_": f'{conf["model"]["_target_"]}.load_from_checkpoint'}, **instantiate_input
    )


def load_prediction_dataset_conf_from_checkpoint(checkpoint_path: str) -> DictConfig:
    """
    Load a dataset conf from a checkpoint path only, inferring it from the dumped hydra conf

    Args:
        checkpoint_path (str):

    Returns:
        Dict

    """
    conf = load_training_conf_from_checkpoint(checkpoint_path)
    return conf.prediction.dataset
