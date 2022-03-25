import logging
from pathlib import Path
from typing import Optional

from classy.utils.experiment import Experiment

logger = logging.getLogger(__name__)


def unzip(zip_path: str, target: Path):
    import zipfile

    """
    Unzip the contents of `zip_path` into `target`.
    """
    logger.debug(f"Unzipping {zip_path} to {target}")
    with zipfile.ZipFile(zip_path) as f:
        f.extractall(target)


def import_zip(zip_path: str, target_path: Optional[str] = None):
    if target_path is None:
        target = Experiment.try_get_experiment_dir()
    else:
        target = Path(target_path)
    unzip(zip_path, target)
