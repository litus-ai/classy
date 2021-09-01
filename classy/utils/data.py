from typing import Tuple, Optional


def split_dataset(
    output_folder: str,
    dataset_path: str,
    validation_split_size: Optional[float] = None,
    test_split_size: Optional[float] = None,
    data_max_split: Optional[int] = None,
) -> Tuple[str, Optional[str], Optional[str]]:
    raise NotImplementedError
