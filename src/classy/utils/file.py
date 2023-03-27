import logging
from pathlib import Path

logger = logging.getLogger(__name__)


CLASSY_HF_MODEL_URL = (
    "https://huggingface.co/{user_name}/{model_name}/resolve/main/model.zip"
)
CLASSY_HF_INFO_URL = (
    "https://huggingface.co/{user_name}/{model_name}/raw/main/info.json"
)
CLASSY_MODELS_CACHE_DIR = ".cache/sunglasses-ai/classy"
CLASSY_MODELS_CACHE_PATH = Path.home() / CLASSY_MODELS_CACHE_DIR
CLASSY_DATE_FORMAT = "%Y-%m-%d %H-%M-%S"


def ensure_dir(path) -> Path:
    """
    Create dir in case it does not exist.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_md5(path: Path):
    """
    Get the MD5 value of a path.
    """
    import hashlib

    with path.open("rb") as fin:
        data = fin.read()
    return hashlib.md5(data).hexdigest()
