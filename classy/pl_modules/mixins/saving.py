import shutil

import pytorch_lightning as pl
from pathlib import Path

from omegaconf import DictConfig

from classy.data.data_drivers import get_data_driver
from classy.data.data_modules import ClassyDataModule
from classy.utils.hydra import fix_paths


class SavingMixin:

    def save_resources_and_update_config(
        self, conf: DictConfig, working_folder: str, experiment_folder: str, data_module: ClassyDataModule
    ):

        working_folder = Path(working_folder)
        experiment_folder = Path(experiment_folder)

        # save examples
        experiment_folder.joinpath("data").mkdir(exist_ok=True)
        get_data_driver(self.task, "jsonl").save(
            data_module.get_test_or_validation_examples(n=5),
            str(experiment_folder.joinpath("data").joinpath("examples.jsonl")),
        )

        # move every paths into "./resources/" and overwrite the config
        Path(experiment_folder / 'resources').mkdir()
        def fix_with_copy_side_effect(path):
            input_path = Path(path)
            assert input_path.exists()
            output_path = experiment_folder / 'resources' / input_path.name
            shutil.copy(input_path, output_path)
            return str(output_path.relative_to(experiment_folder))

        fix_paths(conf.model, check_fn=lambda path: Path(path).exists(), fix_fn=fix_with_copy_side_effect)
        fix_paths(conf.prediction, check_fn=lambda path: Path(path).exists(), fix_fn=fix_with_copy_side_effect)

