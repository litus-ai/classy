from pathlib import Path

import classy
from classy.utils.log import get_project_logger

logger = get_project_logger(__name__)


_optional_requirements_str = """
# classy demo
streamlit==1.5.0
st-annotated-text==2.0.0
# classy serve
fastapi==0.68.1
uvicorn[standard]==0.15.0
# classy train --print
rich==11.1.0
"""
_optional_requirements = {}


for line in _optional_requirements_str.split("\n")[1:-1]:
    line = line.strip()
    if not line.startswith("#"):
        parts = line.split("==")
        d = parts[0]
        if "[" in d:
            d = d[: d.index("[")]
        _optional_requirements[d] = line


def get_optional_requirement(dependant_package: str) -> str:
    if dependant_package not in _optional_requirements:
        logger.error(
            f"Requesting optional dependencies towards {dependant_package}, which however is not listed in classy optional-requirements.txt"
        )
    return _optional_requirements[dependant_package]
