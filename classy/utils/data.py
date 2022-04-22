import itertools
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from classy.data.data_drivers import DataDriver
from classy.utils.log import get_project_logger

logger = get_project_logger(__name__)


def create_data_dir():
    # create data folder
    output_folder = Path("data")
    output_folder.mkdir(exist_ok=True)


def load_dataset(
    path2data_driver: Dict[str, DataDriver],
) -> list:
    dataset_iterator = itertools.chain(
        *[dd.read_from_path(p) for p, dd in path2data_driver.items()]
    )
    return list(dataset_iterator)


def shuffle_dataset(
    path2data_driver: Dict[str, DataDriver],
) -> list:
    samples = load_dataset(path2data_driver)
    np.random.shuffle(samples)
    return samples


def shuffle_and_store_dataset(
    path2data_driver: Dict[str, DataDriver],
    main_data_driver: DataDriver,
    output_path: str,
) -> None:
    samples = shuffle_dataset(path2data_driver)
    main_data_driver.save(samples, output_path)


# TODO: we have to modify this script in order to support the split without loading the whole dataset in memory
def split_dataset(
    path2data_driver: Dict[str, DataDriver],
    main_data_driver: DataDriver,
    main_extension: str,
    output_folder: str,
    validation_split_size: Optional[float] = None,
    test_split_size: Optional[float] = None,
    data_max_split: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[
    Dict[str, DataDriver],
    Optional[Dict[str, DataDriver]],
    Optional[Dict[str, DataDriver]],
]:

    assert (
        sum([validation_split_size or 0.0, test_split_size or 0.0]) > 0.0
    ), "At least one between validation_split_size and test_split_size must be provided with a value > 0"

    # create output folder
    create_data_dir()

    # read samples and shuffle
    if shuffle:
        logger.info("Materializing and shuffling dataset before splitting it")
        samples = shuffle_dataset(path2data_driver)
    else:
        logger.info("Materializing dataset before splitting it")
        samples = load_dataset(path2data_driver)

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
        validation_path = str(output_folder.joinpath(f"validation.{main_extension}"))
        main_data_driver.save(validation_samples, validation_path)

    if test_split_size is not None:
        n_test_samples = min(
            int(len(samples) * test_split_size), data_max_split or len(samples)
        )
        test_samples, training_samples = (
            training_samples[:n_test_samples],
            training_samples[n_test_samples:],
        )
        test_path = str(output_folder.joinpath(f"test.{main_extension}"))
        main_data_driver.save(test_samples, test_path)

    train_path = str(output_folder.joinpath(f"train.{main_extension}"))
    main_data_driver.save(training_samples, train_path)

    train_bundle = {train_path: main_data_driver} if train_path is not None else None
    validation_bundle = (
        {validation_path: main_data_driver} if validation_path is not None else None
    )
    test_bundle = {test_path: main_data_driver} if test_path is not None else None

    return train_bundle, validation_bundle, test_bundle
