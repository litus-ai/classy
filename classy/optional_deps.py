_optional_requirements_str = """
# classy demo
streamlit==1.5.0
st-annotated-text==2.0.0
# classy serve
fastapi==0.68.1
uvicorn[standard]==0.15.0
# reference api
pdoc3==0.10.0
# misc
plotly==5.5.0
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
    return _optional_requirements[dependant_package]


if __name__ == "__main__":
    print(_optional_requirements_str[1:])
