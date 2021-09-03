from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name="classy",
    version="0.1.0",
    description="A simple command line tool to train and test your classification models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    keywords="classy nn ml sunglassesai sunglasses classification ",
    url="https://github.com/sunglasses-ai/classy",
    author="Sunglasses AI",
    author_email="TODO",
    license="Apache",
    packages=find_packages(where="."),
    install_requires=requirements,
    entry_points={"console_scripts": ["classy=classy.scripts.cli.__init__:main"]},
    # include_package_data=True,
    python_requires=">=3.8.0",
    zip_safe=False,
)
