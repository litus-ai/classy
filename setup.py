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


with open("requirements.txt") as f:
    requirements = f.readlines()

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
    packages=find_namespace_packages(include=["hydra_plugins.*"]) + find_packages(include=("classy", "configurations")),
    install_requires=requirements,
    entry_points={"console_scripts": ["classy=classy.scripts.cli.__init__:main"]},
    python_requires=">=3.7.0",
    zip_safe=False,
)
