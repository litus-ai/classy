from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from classy.data.data_drivers import DataDriver
from classy.utils.log import get_project_logger

logger = get_project_logger(__name__)


def create_data_dir():
    # create data folder
    output_folder = Path("data")
    output_folder.mkdir(exist_ok=True)


def load_dataset(
    dataset_path: str,
    data_driver: DataDriver,
) -> list:
    return list(data_driver.read_from_path(dataset_path))


def shuffle_dataset(
    dataset_path: str,
    data_driver: DataDriver,
) -> list:
    samples = load_dataset(dataset_path, data_driver)
    np.random.shuffle(samples)
    return samples


def shuffle_and_store_dataset(
    dataset_path: str,
    data_driver: DataDriver,
    output_path: str,
) -> None:
    samples = shuffle_dataset(dataset_path, data_driver)
    data_driver.save(samples, output_path)


# TODO: we have to modify this script in order to support the split without loading the whole dataset in memory
def split_dataset(
    dataset_path: str,
    data_driver: DataDriver,
    output_folder: str,
    validation_split_size: Optional[float] = None,
    test_split_size: Optional[float] = None,
    data_max_split: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[str, Optional[str], Optional[str]]:

    assert (
        sum([validation_split_size or 0.0, test_split_size or 0.0]) > 0.0
    ), "At least one between validation_split_size and test_split_size must be provided with a value > 0"
    output_extension = dataset_path.split(".")[-1]

    # create output folder
    create_data_dir()

    # read samples and shuffle
    if shuffle:
        logger.info("Materializing and shuffling dataset before splitting it")
        samples = shuffle_dataset(dataset_path, data_driver)
    else:
        logger.info("Materializing dataset before splitting it")
        samples = load_dataset(dataset_path, data_driver)

    # splitting
    training_samples = samples
    train_path, validation_path, test_path = None, None, None

    output_folder = Path(output_folder)

    if validation_split_size is not None:
        n_validation_samples = min(
            int(len(samples) * validation_split_size), data_max_split or len(samples)
        )
        validation_samples, training_samples = (
            training_samples[:n_validation_samples],
            training_samples[n_validation_samples:],
        )
        validation_path = str(output_folder.joinpath(f"validation.{output_extension}"))
        data_driver.save(validation_samples, validation_path)

    if test_split_size is not None:
        n_test_samples = min(
            int(len(samples) * test_split_size), data_max_split or len(samples)
        )
        test_samples, training_samples = (
            training_samples[:n_test_samples],
            training_samples[n_test_samples:],
        )
        test_path = str(output_folder.joinpath(f"test.{output_extension}"))
        data_driver.save(test_samples, test_path)

    train_path = str(output_folder.joinpath(f"train.{output_extension}"))
    data_driver.save(training_samples, train_path)

    return train_path, validation_path, test_path
