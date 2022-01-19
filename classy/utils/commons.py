import itertools
import socket
import subprocess
from typing import Iterable, Optional, Tuple

import numpy as np

from classy.utils.log import get_project_logger

logger = get_project_logger(__name__)


def execute_bash_command(command: str) -> Optional[str]:
    command_result = subprocess.run(command, shell=True, capture_output=True)
    try:
        command_result.check_returncode()
        return command_result.stdout.decode("utf-8")
    except subprocess.CalledProcessError:
        logger.warning(f"failed executing command: {command}")
        logger.warning(f"return code was: {command_result.returncode}")
        logger.warning(f'stdout was: {command_result.stdout.decode("utf-8")}')
        logger.warning(f'stderr code was: {command_result.stderr.decode("utf-8")}')
        return None


def flatten(lst: Iterable[list]) -> list:
    return [_e for sub_l in lst for _e in sub_l]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def grouper(iterable, n):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def split_by_first(text: str, split: str) -> Tuple[str, str]:
    split_idx = text.index(split)
    return text[:split_idx], text[split_idx + len(split) :]


def add_noise_to_value(value: int, noise_param: float):
    noise_value = value * noise_param
    noise = np.random.uniform(-noise_value, noise_value)
    return max(1, value + noise)


def get_local_ip_address() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    address = s.getsockname()[0]
    s.close()
    return address
