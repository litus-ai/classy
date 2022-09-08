import logging
from pathlib import Path
from typing import Callable, Dict

import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf

from classy.pl_modules.base import ClassyPLModule
from classy.utils.hydra import fix_paths
from classy.utils.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


def load_training_conf_from_checkpoint(
    checkpoint_path: str, post_trainer_init: bool = False
) -> DictConfig:
    # find hydra config path
    experiment_folder = Path(checkpoint_path).parent.parent
    # load hydra configs
    conf = OmegaConf.load(f"{experiment_folder}/.hydra/config.yaml")

    # fix paths
    def check_fn(path):
        # check whether path exists relative in the experiment resources folder
        # if it does, fix it
        return (experiment_folder / "resources").joinpath(path).exists()

    fix_paths(
        conf,
        check_fn=check_fn,
        fix_fn=lambda path: str((experiment_folder / "resources").joinpath(path)),
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
    train_dataset_class = conf["data"]["datamodule"]["dataset"]["_target_"]
    if not train_dataset_class.split(".")[-1][
        0
    ].isupper():  # if it is not upper then it is a class method
        train_dataset_class = ".".join(train_dataset_class.split(".")[:-1])

    requires_vocab = hydra.utils.instantiate(
        {"_target_": f"{train_dataset_class}.requires_vocab"}
    )

    # extract and build vocabulary
    vocabulary_path = Path(checkpoint_path).parent.parent / "vocabulary"

    assert (
        not requires_vocab
    ) or vocabulary_path.exists(), f"No vocabulary found at path {vocabulary_path}"

    vocabulary = None
    if vocabulary_path.exists():
        vocabulary = Vocabulary.from_folder(vocabulary_path)

    # prepare instantiate params
    instantiate_input = dict(
        checkpoint_path=checkpoint_path, _recursive_=False, **conf.model
    )
    if vocabulary is not None:
        instantiate_input["vocabulary"] = vocabulary
    instantiate_input["_target_"] = f'{conf.model["_target_"]}.load_from_checkpoint'

    # instantiate and return
    return hydra.utils.instantiate(instantiate_input)


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
