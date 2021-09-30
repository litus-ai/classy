from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name="classy-ml",
    version="0.1.0",
    description="A powerful tool to train and use your classification models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    keywords="classy nn ml sunglassesai classification",
    url="https://github.com/sunglasses-ai/classy",
    author="Sunglasses AI",
    author_email="TODO",
    license="Apache",
    packages=find_packages(include=("classy", "configurations")),
    install_requires=requirements,
    entry_points={"console_scripts": ["classy=classy.scripts.cli.__init__:main"]},
    python_requires=">=3.7.0",
    zip_safe=False,
)
