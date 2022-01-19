import itertools
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import hydra.utils
import omegaconf
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from classy.data.data_drivers import DataDriver, get_data_driver
from classy.utils.data import create_data_dir, shuffle_and_store_dataset, split_dataset
from classy.utils.log import get_project_logger
from classy.utils.vocabulary import Vocabulary

logger = get_project_logger(__name__)


def path_if_exists(path: str, data_driver: DataDriver) -> Optional[str]:
    return path if data_driver.dataset_exists_at_path(path) else None


class ClassyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        task: str,
        dataset_path: str,
        train_dataset: omegaconf.DictConfig,
        validation_dataset: Optional[omegaconf.DictConfig] = None,
        test_dataset: Optional[omegaconf.DictConfig] = None,
        validation_split_size: Optional[float] = None,
        test_split_size: Optional[float] = None,
        max_nontrain_split_size: Optional[int] = None,
        shuffle_dataset: bool = True,
        external_vocabulary_path: Optional[str] = None,
    ):
        super().__init__()
        self.task = task
        self.dataset_path = dataset_path
        self.file_extension = None
        self.data_driver = None

        self.train_path, self.validation_path, self.test_path = None, None, None
        self.train_dataset, self.validation_dataset, self.test_dataset = (
            None,
            None,
            None,
        )
        self.train_dataset_conf = train_dataset
        self.validation_dataset_conf = validation_dataset or train_dataset
        self.test_dataset_conf = test_dataset or validation_dataset or train_dataset

        self.validation_split_size = validation_split_size
        self.test_split_size = test_split_size
        self.max_nontrain_split_size = max_nontrain_split_size
        self.shuffle_dataset = shuffle_dataset

        self.external_vocabulary_path = external_vocabulary_path
        self.vocabulary = None

        # If the data directory exists this run is being resumed from a previous one.
        # If the previous run stored some data we have to load them instead of recomputing
        # the splits or the shuffling.
        if Path("data/").exists():
            files_in_dir = os.listdir("data/")

            # check if the previous run stored a train file
            possible_train_paths = [fp for fp in files_in_dir if "train" in fp]
            if len(possible_train_paths) == 1:
                self.train_path = possible_train_paths[0]

            # check if the previous run stored a validation file
            possible_validation_paths = [
                fp for fp in files_in_dir if "validation" in fp
            ]
            if len(possible_validation_paths) == 1:
                self.validation_path = possible_validation_paths[0]

            # check if the previous run stored a test file
            possible_test_paths = [fp for fp in files_in_dir if "test" in fp]
            if len(possible_test_paths) == 1:
                self.test_path = possible_test_paths[0]

    def get_examples(self, n: int) -> Tuple[str, List]:
        source = self.test_path or self.validation_path
        assert source is not None
        return "test" if self.test_path is not None else "validation", list(
            itertools.islice(self.data_driver.read_from_path(source), n)
        )

    def prepare_data(self) -> None:

        # TODO: we should improve the flow of this code
        if (
            self.train_path is not None
            and self.validation_path is not None
            and self.test_path is not None
        ):
            logger.info(
                "Using train dev and test splits produced by the run being resumed"
            )
        elif Path(
            self.dataset_path
        ).is_dir():  # the user provided a directory containing the datasets
            dir_train_files = [
                fp for fp in os.listdir(self.dataset_path) if "train" in fp
            ]

            assert (
                len(dir_train_files) == 1
            ), f"Expected one file with 'train' in its name, but {len(dir_train_files)} were found in {self.dataset_path}: {dir_train_files}"

            train_file = dir_train_files[0]
            self.file_extension = train_file.split(".")[-1]
            self.data_driver = get_data_driver(self.task, self.file_extension)

            if (
                self.train_path is None
            ):  # does not belong to the train shuffling of a resume run
                self.train_path = path_if_exists(
                    os.path.join(self.dataset_path, f"train.{self.file_extension}"),
                    self.data_driver,
                )

            if self.validation_path is None:
                self.validation_path = path_if_exists(
                    os.path.join(
                        self.dataset_path, f"validation.{self.file_extension}"
                    ),
                    self.data_driver,
                )

            self.test_path = path_if_exists(
                os.path.join(self.dataset_path, f"test.{self.file_extension}"),
                self.data_driver,
            )

            assert (
                self.train_path is not None
            ), f"Cannot find the training file '{self.train_path}'"

            must_shuffle_dataset = (
                self.shuffle_dataset
                and not self.data_driver.dataset_exists_at_path(
                    f"data/train.shuffled.{self.file_extension}"
                )
                and not self.data_driver.dataset_exists_at_path(
                    f"data/dataset.shuffled.{self.file_extension}"
                )
            )

            if must_shuffle_dataset:
                # create data folder
                create_data_dir()
                # shuffle input dataset
                shuffled_dataset_path = f"data/train.shuffled.{self.file_extension}"
                logger.info(
                    f"Shuffling training dataset. The shuffled dataset "
                    f"will be stored at: {os.getcwd()}/{shuffled_dataset_path}"
                )
                shuffle_and_store_dataset(
                    self.train_path, self.data_driver, output_path=shuffled_dataset_path
                )
                self.train_path = shuffled_dataset_path

            if self.validation_path is None:
                logger.info(
                    f"Validation dataset not found: splitting the training dataset "
                    f"(split_size: {1 - self.validation_split_size} / {self.validation_split_size})"
                    f"enforcing a maximum of {self.max_nontrain_split_size} instances on validation dataset"
                )

                # if we must split the shuffled train dataset in two, then we must change its name
                if must_shuffle_dataset:
                    shuffled_dataset_path = (
                        f"data/dataset.shuffled.{self.file_extension}"
                    )
                    os.rename(self.train_path, shuffled_dataset_path)
                    self.train_path = shuffled_dataset_path  # will be modified in the next lines of code

                self.train_path, self.validation_path, _ = split_dataset(
                    self.train_path,
                    self.data_driver,
                    "data/",  # hydra takes care of placing this folder within the appropriate folder
                    validation_split_size=self.validation_split_size,
                    data_max_split=self.max_nontrain_split_size,
                    shuffle=False,
                )
                logger.info(
                    f"Storing the newly created datasets at '{self.train_path}' and '{self.validation_path}'"
                )

        else:  # the user provided just one file that must be split in train, dev and test
            self.file_extension = self.dataset_path.split(".")[-1]
            self.data_driver = get_data_driver(self.task, self.file_extension)

            if self.shuffle_dataset and not self.data_driver.dataset_exists_at_path(
                f"data/dataset.shuffled.{self.file_extension}"
            ):
                # create data folder
                create_data_dir()
                # shuffle training dataset
                shuffled_dataset_path = f"data/dataset.shuffled.{self.file_extension}"
                logger.info(
                    f"Shuffling input dataset. The shuffled dataset "
                    f"will be stored at: {os.getcwd()}/{shuffled_dataset_path}"
                )
                shuffle_and_store_dataset(
                    self.dataset_path,
                    self.data_driver,
                    output_path=shuffled_dataset_path,
                )
                self.dataset_path = shuffled_dataset_path

            # splitting dataset in train, validation and test
            logger.info(
                "Splitting the dataset in train, validation and test. "
                f"(split_size: {1 - self.validation_split_size - self.test_split_size} "
                f"/ {self.validation_split_size}, {self.test_split_size}) "
                f"enforcing a maximum of {self.max_nontrain_split_size} instances on non-train splits"
            )

            self.train_path, self.validation_path, self.test_path = split_dataset(
                self.dataset_path,
                self.data_driver,
                "data/",  # hydra takes care of placing this folder within the appropriate folder
                validation_split_size=self.validation_split_size,
                test_split_size=self.test_split_size,
                data_max_split=self.max_nontrain_split_size,
                shuffle=False,
            )

            logger.info(
                f"Storing the newly created datasets at '{self.train_path}', "
                f"'{self.validation_path}'and '{self.test_path}'"
            )

        # todo: can we improve it?
        if Path("vocabulary/").exists():
            logger.info("Loading vocabulary from previous run")
            self.vocabulary = Vocabulary.from_folder("vocabulary/")
        elif self.external_vocabulary_path is not None:
            logger.info("Loading vocabulary from external passed directory")
            self.vocabulary = Vocabulary.from_folder(self.external_vocabulary_path)
            self.vocabulary.save("vocabulary")
        else:
            self.vocabulary = hydra.utils.instantiate(
                self.train_dataset_conf,
                path=self.train_path,
                data_driver=self.data_driver,
            ).vocabulary
            if self.vocabulary is not None:
                self.vocabulary.save("vocabulary")

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == "fit":
            self.train_dataset = hydra.utils.instantiate(
                self.train_dataset_conf,
                path=self.train_path,
                data_driver=self.data_driver,
                vocabulary=self.vocabulary,
            )
            self.validation_dataset = hydra.utils.instantiate(
                self.validation_dataset_conf,
                path=self.validation_path,
                data_driver=self.data_driver,
                vocabulary=self.vocabulary,
            )
        if stage == "test":
            self.test_dataset = hydra.utils.instantiate(
                self.test_dataset_conf,
                path=self.test_path,
                data_driver=self.data_driver,
                vocabulary=self.vocabulary,
            )

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.train_dataset, batch_size=None, num_workers=0)

    def val_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.validation_dataset, batch_size=None, num_workers=0)

    def test_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.test_dataset, batch_size=None, num_workers=0)
