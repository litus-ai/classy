import collections
from typing import Dict, Optional, Tuple, Union

import hydra
from omegaconf import DictConfig, ListConfig

from classy.data.data_drivers import DataDriver, get_data_driver
from classy.utils.log import get_project_logger

logger = get_project_logger(__name__)


def load_bundle(
    bundle_conf: Optional[Union[str, Dict[str, str]]],
    task: str,
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
    elif type(bundle_conf) == ListConfig:
        file_extensions = [path.split(".")[-1] for path in bundle_conf]
        bundle_store = {
            hydra.utils.to_absolute_path(path): get_data_driver(task, file_extension)
            for path, file_extension in zip(bundle_conf, file_extensions)
        }
        if compute_main_extension:
            main_extension = collections.Counter(file_extensions).most_common(1)[0][0]
    elif type(bundle_conf) == DictConfig:
        bundle_store = {
            hydra.utils.to_absolute_path(path): get_data_driver(task, file_extension)
            for path, file_extension in bundle_conf.items()
        }
        if compute_main_extension:
            main_extension = collections.Counter(bundle_conf.values()).most_common(1)[
                0
            ][0]
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
