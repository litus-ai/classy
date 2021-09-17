from pathlib import Path
from typing import Dict

import hydra
from omegaconf import OmegaConf, DictConfig

from classy.pl_modules.base import ClassyPLModule
from classy.utils.vocabulary import Vocabulary

import logging

logger = logging.getLogger(__name__)


def load_training_conf_from_checkpoint(checkpoint_path: str, post_trainer_init: bool = False) -> DictConfig:
    # find hydra config path
    hydra_config_path = "/".join(checkpoint_path.split("/")[:-2])
    # load hydra config
    conf_file = "config_post_trainer_init.yaml" if post_trainer_init else "config.yaml"
    conf = OmegaConf.load(f"{hydra_config_path}/.hydra/{conf_file}".lstrip("/"))
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

    # extract and build vocabulary
    vocabulary_path = Path(checkpoint_path).parent.parent / "vocabulary"
    if not vocabulary_path.exists():
        logger.warning(
            "No vocabulary found for the selected checkpoint. In the current version of classy this is "
            "correct only if the task is 'qa'"
        )
        vocabulary = None
    else:
        vocabulary = Vocabulary.from_folder(vocabulary_path)

    # instantiate and return
    instantiate_input = dict(checkpoint_path=checkpoint_path)
    if vocabulary is not None:
        instantiate_input["vocabulary"] = vocabulary

    return hydra.utils.instantiate(
        {"_target_": f'{conf["model"]["_target_"]}.load_from_checkpoint'}, **instantiate_input
    )


def load_prediction_dataset_conf_from_checkpoint(checkpoint_path: str) -> Dict:
    """
    Load a dataset conf from a checkpoint path only, inferring it from the dumped hydra conf

    Args:
        checkpoint_path (str):

    Returns:
        Dict

    """
    conf = load_training_conf_from_checkpoint(checkpoint_path)
    return dict(conf.prediction.dataset)
