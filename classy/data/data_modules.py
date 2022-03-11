import collections
import itertools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import hydra.utils
import omegaconf
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from classy.data.data_drivers import DataDriver, get_data_driver
from classy.utils.data import create_data_dir, shuffle_and_store_dataset, split_dataset
from classy.utils.log import get_project_logger
from classy.utils.vocabulary import Vocabulary

logger = get_project_logger(__name__)


def path_if_exists(path: str, data_driver: DataDriver) -> Optional[str]:
    return path if data_driver.dataset_exists_at_path(path) else None


@dataclass
class TrainCoordinates:
    main_file_extension: str
    main_data_driver: DataDriver
    train_bundle: Dict[str, DataDriver]
    validation_bundle: Optional[Dict[str, DataDriver]]
    test_bundle: Optional[Dict[str, DataDriver]]


def load_coordinates(coordinates_path: str, task: str) -> TrainCoordinates:
    """
    Computes the train coordinates of a training process.
    Args:
        coordinates_path:
            a path to
                - a file containing the training coordinates (check the documentation for more info)
                - a single file containing the whole dataset to be split in train / dev
                - directory containing two (three) files for train, validation (and test)
        task:
            one of the supported tasks in classy (e.g. sentence-pair)
    Returns:
        train_coordinates (TrainCoordinates): the train_coordinates containing
         all the info on the datasets involved in the training.
    """

    train_coordinates = TrainCoordinates(None, None, None, None, None)

    # If the "data" directory exists (and so this a resume from a previous run)
    # or the "coordinates_path" point to a directory we have to retrieve the data
    # from there.
    if Path("data/").exists() or Path(coordinates_path).is_dir():

        def scan_dir_for_file(
                dir_path: str, file_name: str
        ) -> Optional[Tuple[str, str, DataDriver]]:
            files_in_dir = os.listdir(dir_path)
            matching_files = [fp for fp in files_in_dir if file_name in fp]

            if len(matching_files) == 1:
                matching_file = matching_files[0]
                file_extension = matching_file.split(".")[-1]
                data_driver = get_data_driver(task, file_extension)
                return f"{dir_path}/{matching_file}", file_extension, data_driver

            return None

        previous_run_dir = "data/"

        # check if a previous run stored a train file
        train_scan_output = scan_dir_for_file(previous_run_dir, "train")
        # if not retrieve it from the directory at the coordinates path
        if train_scan_output is None:
            train_scan_output = scan_dir_for_file(coordinates_path, "train")

        assert train_scan_output is not None, "The training file could not be found."

        # Load training data
        train_file, train_file_extension, train_data_driver = train_scan_output
        train_coordinates.main_file_extension = train_file_extension
        train_coordinates.main_data_driver = train_data_driver
        train_coordinates.train_bundle = {train_file: train_data_driver}

        # check if the previous run stored a validation file
        validation_scan_output = scan_dir_for_file(previous_run_dir, "validation")
        # if not retrieve it from the directory at the coordinates path
        if validation_scan_output is None:
            validation_scan_output = scan_dir_for_file(coordinates_path, "validation")
        if validation_scan_output is not None:
            validation_file, _, validation_data_driver = validation_scan_output
            train_coordinates.validation_bundle = {
                validation_file: validation_data_driver
            }

        # check if the previous run stored a test file
        test_scan_output = scan_dir_for_file(previous_run_dir, "test")
        # if not retrieve it from the directory at the coordinates path
        if test_scan_output is None:
            test_scan_output = scan_dir_for_file(coordinates_path, "test")
        if test_scan_output is not None:
            test_file, _, test_data_driver = test_scan_output
            train_coordinates.test_bundle = {test_file: test_data_driver}

    # If the coordinates_path points to a file, it is either a yaml
    # file containing the paths the datasets or a single file that
    # we have to split in train and dev.
    elif Path(coordinates_path).is_file():

        if coordinates_path.split(".")[-1] == "yaml":

            def load_bundle(
                    bundle_conf: Optional[Union[str, Dict[str, str]]],
                    compute_main_extension: bool = False,
            ) -> Optional[Union[Dict[str, DataDriver], Tuple[Dict[str, DataDriver], str]]]:
                if bundle_conf is None:
                    return None

                main_extension = None
                if type(bundle_conf) == str:
                    file_extension = bundle_conf.split(".")[-1]
                    bundle_store = {
                        hydra.utils.to_absolute_path(bundle_conf): get_data_driver(
                            task, file_extension
                        )
                    }
                    if compute_main_extension:
                        main_extension = file_extension
                elif type(bundle_conf) == list:
                    file_extensions = [path.split(".")[-1] for path in bundle_conf]
                    bundle_store = {
                        hydra.utils.to_absolute_path(path): file_extension
                        for path, file_extension in zip(bundle_conf, file_extensions)
                    }
                    if compute_main_extension:
                        main_extension = collections.Counter(file_extensions).most_common(
                            1
                        )[0][0]
                elif type(bundle_conf) == dict:
                    bundle_store = {
                        hydra.utils.to_absolute_path(path): get_data_driver(
                            task, file_extension
                        )
                        for path, file_extension in bundle_conf
                    }
                    if compute_main_extension:
                        main_extension = collections.Counter(
                            bundle_store.values()
                        ).most_common(1)[0][0]
                else:
                    logger.error(
                        "The value of the dataset in the coordinates file "
                        "must be either a string indicating the dataset, a "
                        "list of string or  a dict path -> file_extension"
                    )
                    raise NotImplementedError

                if main_extension is not None:
                    return bundle_store, main_extension
                else:
                    return bundle_store

            coordinates_dict = OmegaConf.load(coordinates_path)

            assert (
                "train_dataset" in coordinates_dict
            ), "The coordinates file must specify the 'train_dataset' field"

            # assign the main file extension if specified in the config
            train_coordinates.main_file_extension = coordinates_dict.get(
                "main_file_extension", None
            )

            # train_bundle
            if train_coordinates.main_file_extension is None:
                (
                    train_coordinates.train_bundle,
                    train_coordinates.main_file_extension,
                ) = load_bundle(
                    coordinates_dict.get("train_dataset"), compute_main_extension=True
                )
            else:
                train_coordinates.train_bundle = load_bundle(
                    coordinates_dict.get("train_dataset")
                )
            train_coordinates.main_data_driver = get_data_driver(
                task, train_coordinates.main_file_extension
            )

            # validation_bundle
            train_coordinates.validation_bundle = load_bundle(
                coordinates_dict.get("validation_dataset")
            )

            # test_bundle
            train_coordinates.test_bundle = load_bundle(
                coordinates_dict.get("test_dataset")
            )

        else:
            # just one file that will later be split in train and dev
            train_coordinates.main_file_extension = coordinates_path.split(".")[-1]
            train_coordinates.main_data_driver = get_data_driver(
                task, train_coordinates.main_file_extension
            )
            train_coordinates.train_bundle = {
                coordinates_path: train_coordinates.main_data_driver
            }
    else:
        raise NotImplementedError

    return train_coordinates


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

        self.train_coordinates: TrainCoordinates = None
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
        source = (
            self.train_coordinates.test_bundle
            or self.train_coordinates.validation_bundle
        )
        dataset_path, data_driver = list(source.items())[0]
        assert source is not None
        return (
            "test" if self.train_coordinates.test_bundle is not None else "validation",
            list(itertools.islice(data_driver.read_from_path(dataset_path), n)),
        )

    def build_vocabulary(self) -> None:
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
                path=self.train_coordinates.train_bundle,
            ).vocabulary
            if self.vocabulary is not None:
                self.vocabulary.save("vocabulary")

    def prepare_data(self) -> None:
        train_coordinates = load_coordinates(self.dataset_path, self.task)

        # Check it the training dataset has been split or not
        # If so check that it is the only dataset in the train_bundle
        if train_coordinates.main_data_driver.dataset_exists_at_path(
            f"data/train.shuffled.{train_coordinates.main_file_extension}"
        ):
            assert (
                len(train_coordinates.train_bundle) == 1
                and list(train_coordinates.train_bundle.keys())[0]
                == f"data/train.shuffled.{train_coordinates.main_file_extension}"
            ), (
                f"If the train.shuffled.{train_coordinates.main_file_extension} "
                f"already exists it must be the only training dataset at this point."
            )

        must_shuffle_dataset = (
            self.shuffle_dataset
            and not train_coordinates.main_data_driver.dataset_exists_at_path(
                f"data/train.shuffled.{train_coordinates.main_file_extension}"
            )
        )

        if must_shuffle_dataset:
            # create data folder
            create_data_dir()
            # shuffle input dataset
            shuffled_dataset_path = (
                f"data/train.shuffled.{train_coordinates.main_file_extension}"
            )
            logger.info(
                f"Shuffling training dataset. The shuffled dataset "
                f"will be stored at: {os.getcwd()}/{shuffled_dataset_path}"
            )
            shuffle_and_store_dataset(
                train_coordinates.train_bundle,
                train_coordinates.main_data_driver,
                output_path=shuffled_dataset_path,
            )
            train_coordinates.train_bundle = {
                shuffled_dataset_path: train_coordinates.main_data_driver
            }

        if train_coordinates.validation_bundle is None:
            logger.info(
                f"Validation dataset not found: splitting the training dataset "
                f"(split_size: {1 - self.validation_split_size} / {self.validation_split_size})"
                f"enforcing a maximum of {self.max_nontrain_split_size} instances on validation dataset"
            )

            # if we must split the shuffled train dataset in two, then we must change its name
            if must_shuffle_dataset:
                shuffled_dataset_path = f"data/dataset.shuffled.{self.file_extension}"
                os.rename(self.train_path, shuffled_dataset_path)
                train_coordinates.train_bundle = {
                    shuffled_dataset_path: train_coordinates.main_data_driver
                }

            # split and assign
            (
                train_coordinates.train_bundle,
                train_coordinates.validation_bundle,
                _,
            ) = split_dataset(
                train_coordinates.train_bundle,
                train_coordinates.main_data_driver,
                train_coordinates.main_file_extension,
                "data/",  # hydra takes care of placing this folder within the appropriate folder
                validation_split_size=self.validation_split_size,
                data_max_split=self.max_nontrain_split_size,
                shuffle=False,
            )
            logger.info(
                f"Storing the newly created datasets at '{self.train_path}' and '{self.validation_path}'"
            )

        self.train_coordinates = train_coordinates

        # build vocabulary
        self.build_vocabulary()

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == "fit":
            self.train_dataset = hydra.utils.instantiate(
                self.train_dataset_conf,
                path=self.train_coordinates.train_bundle,
                vocabulary=self.vocabulary,
            )
            self.validation_dataset = hydra.utils.instantiate(
                self.validation_dataset_conf,
                path=self.train_coordinates.validation_bundle,
                vocabulary=self.vocabulary,
            )
        if stage == "test":
            self.test_dataset = hydra.utils.instantiate(
                self.test_dataset_conf,
                path=self.train_coordinates.test_bundle,
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
