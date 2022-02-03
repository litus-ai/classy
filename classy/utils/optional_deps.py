from pathlib import Path

import classy
from classy.utils.log import get_project_logger

logger = get_project_logger(__name__)


optional_requirements = {}


with open(Path(classy.__file__).parent.parent / "requirements-optional.txt") as f:
    for line in f:
        line = line.strip()
        if not line.startswith("#"):
            parts = line.split("==")
            d = parts[0]
            if "[" in d:
                d = d[: d.index("[")]
            optional_requirements[d] = line


def get_optional_requirement(dependant_package: str) -> str:
    if dependant_package not in optional_requirements:
        logger.error(
            f"Requesting optional dependencies towards {dependant_package}, which however is not listed in classy optional-requirements.txt"
        )
    return optional_requirements[dependant_package]
