import os
import shutil
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig

from classy.data.data_drivers import get_data_driver
from classy.data.data_modules import ClassyDataModule
from classy.utils.hydra import fix_paths


class SavingMixin:
    def save_resources_and_update_config(
        self,
        conf: DictConfig,
        working_folder: str,
        experiment_folder: str,
        data_module: ClassyDataModule,
    ):

        working_folder = Path(working_folder)
        experiment_folder = Path(experiment_folder)

        # save examples
        source, examples = data_module.get_examples(n=5)
        experiment_folder.joinpath("data").mkdir(exist_ok=True)
        get_data_driver(self.task, "jsonl").save(
            examples, str(experiment_folder / "data" / f"examples-{source}.jsonl")
        )

        # move every paths into "./resources/" and overwrite the config
        Path(experiment_folder / "resources").mkdir()

        def fix_with_copy_side_effect(path):
            input_path = Path(path)
            assert input_path.exists()
            output_path = (
                experiment_folder / "resources" / input_path.relative_to(working_folder)
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if os.path.isdir(input_path):
                shutil.copytree(input_path, output_path)
            else:
                shutil.copy(input_path, output_path)
            return str(output_path.relative_to(experiment_folder))

        fix_paths(
            conf.model,
            check_fn=lambda path: Path(path).exists(),
            fix_fn=fix_with_copy_side_effect,
        )
        fix_paths(
            conf.prediction,
            check_fn=lambda path: Path(path).exists(),
            fix_fn=fix_with_copy_side_effect,
        )
        if "evaluation" in conf:
            fix_paths(
                conf.evaluation,
                check_fn=lambda path: Path(path).exists(),
                fix_fn=fix_with_copy_side_effect,
            )
