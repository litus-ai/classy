import random
from pathlib import Path
from typing import Tuple, Optional

from classy.data.data_drivers import DataDriver
from classy.utils.log import get_project_logger

logger = get_project_logger(__name__)


def split_dataset(
    dataset_path: str,
    data_driver: DataDriver,
    output_folder: str,
    validation_split_size: Optional[float] = None,
    test_split_size: Optional[float] = None,
    data_max_split: Optional[int] = None,
) -> Tuple[str, Optional[str], Optional[str]]:

    assert sum([validation_split_size or 0.0, test_split_size or 0.0]) > 0.0, 'At least one between validation_split_size and test_split_size must be provided with a value > 0'
    output_extension = dataset_path.split('.')[-1]

    # create output folder
    output_folder = Path(output_folder)
    output_folder.mkdir()

    # read samples and shuffle
    logger.info('Materializing and shuffling dataset before splitting it')
    samples = list(data_driver.read_from_path(dataset_path))
    random.shuffle(samples)

    # splitting

    training_samples = samples
    train_path, validation_path, test_path = None, None, None

    if validation_split_size is not None:
        n_validation_samples = min(int(len(samples) * validation_split_size), data_max_split or len(samples))
        validation_samples, training_samples = training_samples[: n_validation_samples], training_samples[n_validation_samples:]
        validation_path = str(output_folder.joinpath(f'validation.{output_extension}'))
        data_driver.save(validation_samples, validation_path)

    if test_split_size is not None:
        n_test_samples = min(int(len(samples) * test_split_size), data_max_split or len(samples))
        test_samples, training_samples = training_samples[: n_test_samples], training_samples[n_test_samples:]
        test_path = str(output_folder.joinpath(f'test.{output_extension}'))
        data_driver.save(test_samples, test_path)

    train_path = str(output_folder.joinpath(f'train.{output_extension}'))
    data_driver.save(training_samples, train_path)

    return train_path, validation_path, test_path
