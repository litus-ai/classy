import collections
import re

from setuptools import setup, find_packages, find_namespace_packages

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# release markers:
#   X.Y
#   X.Y.Z      # For bugfix releases
#
# pre-release markers:
#   X.YaN      # Alpha release
#   X.YbN      # Beta release
#   X.YrcN     # Release Candidate
#   X.Y.ZdevD  # Nightly Builds
#   X.Y        # Final release

# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import classy whilst setting up.

VERSION = {}  # type: ignore
with open("classy/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)


# read requirements
requirements = []
with open("requirements.txt") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)


# read extra requirements
extra_requirements = collections.defaultdict(set)
with open("extra-requirements.txt") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            k, tags = line, set()
            if '\t' in k:
                k, _tags = k.split('\t')
                tags.update(_tag.strip() for _tag in _tags.split(','))
            tags.add(re.split("[<=>]", k)[0])
            for tag in tags:
                extra_requirements[tag].add(k)
extra_requirements["all"] = set([req for reqs in extra_requirements.values() for req in reqs])

setup(
    name="classy-core",
    version=VERSION["VERSION"],
    description="A powerful tool to train and use your classification models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="classy nn ml sunglassesai classification",
    url="https://github.com/sunglasses-ai/classy",
    author="Classy Team @ Sunglasses AI",
    author_email="classy@sunglasses.ai",
    license="Apache",
    packages=find_packages(),
    package_data={"configurations": ["*.yaml", "*/*.yaml"]},
    install_requires=requirements,
    extras_require=extra_requirements,
    entry_points={"console_scripts": ["classy=classy.scripts.cli.__init__:main"]},
    python_requires=">=3.8.0",
    zip_safe=False,
)
