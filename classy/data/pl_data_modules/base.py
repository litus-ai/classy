import os
from typing import Optional, Union, List, Dict

import hydra.utils
import omegaconf
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from classy.data.readers import get_reader
from classy.utils.data import split_dataset

import logging

from classy.utils.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


class ClassyDataModule(pl.LightningDataModule):
    def __init__(self, conf: omegaconf.DictConfig):
        super().__init__()
        self.conf = conf
        self.dataset_path = self.conf.data.dataset_path
        self.files_extension = None
        self.file_reader = None

        self.train_path = None
        self.validation_path = None
        self.test_path = None

        self.features_vocabulary: Vocabulary = None
        self.labels_vocabulary: Vocabulary = None

        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:

        if os.path.isdir(self.dataset_path):
            dir_files = [fp for fp in os.listdir(self.dataset_path) if "train" in fp]

            assert len(dir_files) == 1, "Found more than one file with 'train' in their name"  # todo: expand

            self.files_extension = dir_files[0].split(".")[-1]

            self.train_path = os.path.join(self.conf.data.dataset_path, f"train.{self.files_extension}")
            self.validation_path = os.path.join(self.conf.data.dataset_path, f"validation.{self.files_extension}")
            self.test_path = os.path.join(self.conf.data.dataset_path, f"test.{self.files_extension}")

            assert os.path.exists(self.train_path), f"Cannot find the training file 'train.{self.files_extension}'"
            assert os.path.exists(self.test_path), f"Cannot find the training file 'test.{self.files_extension}'"

            if not os.path.exists(self.validation_path):
                logger.info(
                    f"Validation dataset not found: splitting the training dataset "
                    f"(split_size: {1 - self.conf.data.validation_split_size} / {self.conf.data.validation_split_size})"
                )
                self.train_path, self.validation_path, _ = split_dataset(
                    self.train_path,
                    validation_split_size=self.conf.data.validation_split_size,
                    data_max_split=self.conf.data.data_max_split,
                )
                logger.info(f"Storing the newly created datasets at '{self.train_path}' and '{self.validation_path}'")

        else:

            logger.info(
                "Splitting the dataset in train, validation and test. "
                f"(split_size: {1 - self.conf.data.validation_split_size - self.conf.data.test_split_size} "
                f"/ {self.conf.data.validation_split_size}, {self.conf.data.test_split_size})"
            )

            self.files_extension = self.train_path.split(".")[-1]

            self.train_path, self.validation_path, self.test_path = split_dataset(
                self.train_path,
                validation_split_size=self.conf.data.validation_split_size,
                test_split_size=self.conf.data.test_split_size,
                data_max_split=self.conf.data.data_max_split,
            )

            logger.info(
                f"Storing the newly created datasets at '{self.train_path}', "
                f"'{self.validation_path}'and '{self.test_path}'"
            )

        # todo: can we improve it?
        self.file_reader = get_reader(self.conf.task_type, self.files_extension)
        train_dataset = hydra.utils.instantiate(
            self.conf.data.train_dataset, path=self.train_path, file_reader=self.file_reader
        )
        self.features_vocabulary = train_dataset.features_vocabulary
        self.labels_vocabulary = train_dataset.labels_vocabulary

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == "fit":
            self.train_dataset = hydra.utils.instantiate(
                self.conf.data.train_dataset,
                path=self.train_path,
                file_reader=self.file_reader,
                features_vocabulary=self.features_vocabulary,
                labels_vocabulary=self.labels_vocabulary,
            )
            self.validation_dataset = hydra.utils.instantiate(
                self.conf.data.validation_dataset,
                path=self.validation_path,
                file_reader=self.file_reader,
                features_vocabulary=self.features_vocabulary,
                labels_vocabulary=self.labels_vocabulary,
            )
        if stage == "test":
            self.test_dataset = hydra.utils.instantiate(
                self.conf.data.test_dataset,
                path=self.test_path,
                file_reader=self.file_reader,
                features_vocabulary=self.features_vocabulary,
                labels_vocabulary=self.labels_vocabulary,
            )

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.train_dataset, batch_size=None, num_workers=1)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.validation_dataset, batch_size=None, num_workers=1)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.test_dataset, batch_size=None, num_workers=1)
