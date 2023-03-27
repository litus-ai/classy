import functools
import logging
import shlex
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf

from classy.pl_modules.base import ClassyPLModule
from classy.scripts.cli.utils import DRY_MODEL
from classy.utils.hydra import fix_paths
from classy.utils.vocabulary import Vocabulary

from ..scripts.cli.train import get_parser, training_args_to_cfg_in_folder

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=500_000)
def load_training_conf(
    checkpoint_path: str, dry_model_configuration: Optional[str] = None
) -> DictConfig:
    """
    Load a training configuration from one of the following:
      * a checkpoint path only (infer the model to load from the dumped hydra conf)
      * a dry model configuration

    Args:
        checkpoint_path (str):
        dry_model_configuration (str):

    Returns:
        DictConfig

    """

    assert bool(checkpoint_path != DRY_MODEL) != bool(
        dry_model_configuration is not None
    ), "Either a checkpoint or a configuration for a dry model must be provided"

    if checkpoint_path != DRY_MODEL:
        return _load_training_conf_from_checkpoint_path(checkpoint_path)
    else:
        return _load_training_conf_from_dry_model_configuration(dry_model_configuration)


def _load_training_conf_from_checkpoint_path(
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


def _load_training_conf_from_dry_model_configuration(
    dry_model_configuration: str,
) -> DictConfig:
    # split args as we if we were in the shell
    shell_args = shlex.split(dry_model_configuration)

    # check none dataset
    if shell_args[1] == "none":
        logger.warning(
            "dataset_path provided is none. This will result in an error if the model specified requires a vocabulary"
        )

    # add -n (which should not be present)
    found_name_arg = False
    for i in range(len(shell_args)):
        if shell_args[i] == "-n":
            logger.info(
                f"Found name parameter: {shell_args[i]} {shell_args[i + 1]}. Note that this parameter can be omitted"
            )
            found_name_arg = True
        if shell_args[i] == "-d":
            logger.error(
                f"Found device parameter in the dry-model-configuration: {shell_args[i]} {shell_args[i + 1]}. "
                f"This is unsupported on purpose, as the device should be set through the outer script parameter "
                f"(e.g., 'classy evaluate [..] -d 0 [...] --dry-model-configuration [...]')"
            )
            raise ValueError(
                f"Found device parameter in the dry-model-configuration: {shell_args[i]} {shell_args[i + 1]}"
            )

    if not found_name_arg:
        shell_args = shell_args[:2] + ["-n", "dry-model"] + shell_args[2:]

    # add -d cpu
    shell_args += ["-d", "cpu"]

    # instantiate training parser and parse args
    train_parser = get_parser()
    train_args = train_parser.parse_args(shell_args)

    # train args => configuration
    with tempfile.TemporaryDirectory() as temporary_directory:
        # write configuration to folder
        training_args_to_cfg_in_folder(train_args, temporary_directory)
        # load it
        with hydra.initialize_config_dir(
            config_dir=temporary_directory, job_name="dry", version_base=None
        ):
            conf = hydra.compose(config_name="run", return_hydra_config=True)

    # return
    return conf


def load_classy_module_and_prediction_dataset_conf(
    checkpoint_path: str, dry_model_configuration: Optional[str] = None
) -> Tuple[ClassyPLModule, DictConfig]:
    """
    Load a PL module from one of the following:
      * a checkpoint path only (infer the model to load from the dumped hydra conf)
      * a dry model configuration

    Args:
        checkpoint_path (str):
        dry_model_configuration (str):

    Returns:
        tuple(ClassyPLModule, DictConfig)

    """

    training_conf = load_training_conf(checkpoint_path, dry_model_configuration)

    if checkpoint_path != DRY_MODEL:
        return _load_classy_module_and_prediction_dataset_conf_from_checkpoint_path(
            checkpoint_path, training_conf=training_conf
        )
    else:
        return _load_classy_module_and_prediction_dataset_conf_from_dry_model_configuration(
            dry_model_configuration, training_conf=training_conf
        )


def _load_classy_module_and_prediction_dataset_conf_from_checkpoint_path(
    checkpoint_path: str, training_conf: DictConfig
) -> Tuple[ClassyPLModule, DictConfig]:
    # extract prediction dataset conf
    prediction_dataset_conf = training_conf.prediction.dataset

    # check if the model requires a vocab
    train_dataset_class = training_conf["data"]["datamodule"]["dataset"]["_target_"]
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
        checkpoint_path=checkpoint_path, _recursive_=False, **training_conf.model
    )
    if vocabulary is not None:
        instantiate_input["vocabulary"] = vocabulary
    instantiate_input[
        "_target_"
    ] = f'{training_conf.model["_target_"]}.load_from_checkpoint'

    # instantiate model
    model = hydra.utils.instantiate(instantiate_input)

    # return
    return model, prediction_dataset_conf


def _load_classy_module_and_prediction_dataset_conf_from_dry_model_configuration(
    dry_model_configuration: str, training_conf: DictConfig
):
    # extract training dataset class (needed to check if a vocabulary is needed)
    train_dataset_class = training_conf.data.datamodule.dataset["_target_"]
    if not train_dataset_class.split(".")[-1][
        0
    ].isupper():  # if it is not upper then it is a class method
        train_dataset_class = ".".join(train_dataset_class.split(".")[:-1])

    # instantiate model
    if hydra.utils.instantiate({"_target_": f"{train_dataset_class}.requires_vocab"}):
        pl_data_module = hydra.utils.instantiate(
            training_conf.data.datamodule,
            external_vocabulary_path=getattr(
                training_conf.data, "vocabulary_dir", None
            ),
            _recursive_=False,
        )
        pl_data_module.prepare_data()

        # main module declaration
        pl_module_init = {"_recursive_": False}
        if pl_data_module.vocabulary is not None:
            pl_module_init["vocabulary"] = pl_data_module.vocabulary
        pl_module = hydra.utils.instantiate(training_conf.model, **pl_module_init)
    else:
        pl_module = hydra.utils.instantiate(training_conf.model, _recursive_=False)

    # return
    return pl_module, training_conf.prediction.dataset
