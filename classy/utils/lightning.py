from pathlib import Path
from typing import Dict

import hydra
from omegaconf import OmegaConf

from classy.pl_modules.base import ClassyPLModule
from classy.utils.vocabulary import Vocabulary


def load_classy_module_from_checkpoint(checkpoint_path: str) -> ClassyPLModule:
    """
    Load a PL module from a checkpoint path only. Infer the model to load from the dumped hydra conf

    Args:
        checkpoint_path (str):

    Returns:
        pl.LightningModule

    """

    # find hydra config path
    hydra_config_path = "/".join(checkpoint_path.split("/")[:-2])

    # load hydra config
    conf = OmegaConf.load(f"{hydra_config_path}/.hydra/config.yaml")

    # extract and build vocabulary
    vocabulary_path = Path(checkpoint_path).parent.parent / "vocabulary"
    assert vocabulary_path.exists(), f"No vocabulary found at path {vocabulary_path}"
    vocabulary = Vocabulary.from_folder(vocabulary_path)

    # instantiate and return
    return hydra.utils.instantiate(
        {"_target_": f'{conf["model"]["_target_"]}.load_from_checkpoint'},
        checkpoint_path=checkpoint_path,
        vocabulary=vocabulary,
    )


def load_prediction_dataset_conf_from_checkpoint(checkpoint_path: str) -> Dict:
    """
    Load a dataset conf from a checkpoint path only, inferring it from the dumped hydra conf

    Args:
        checkpoint_path (str):

    Returns:
        Dict

    """

    # find hydra config path
    hydra_config_path = "/".join(checkpoint_path.split("/")[:-2])

    # load hydra config
    conf = OmegaConf.load(f"{hydra_config_path}/.hydra/config.yaml")

    # instantiate and return
    return dict(conf.prediction.dataset)
