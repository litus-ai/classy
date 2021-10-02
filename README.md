<div align="center">
    <br>
    <img alt="classy logo" src="TODO" width="400"/>
    <p>
    A Pytorch-based library for fast prototyping and sharing of deep neural network models. 
    </p>
    <hr/>
</div>
<p align="center">
<!-- // REMOVE
    <a href="https://github.com/sunglasses-ai/classy/actions">
        <img alt="CI" src="https://github.com/sunglasses-ai/classy/workflows/CI/badge.svg?event=push&branch=main">
    </a>
    <a href="https://github.com/sunglasses-ai/classy/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/sunglasses-ai/classy.svg?color=blue&cachedrop">
    </a>
    <a href="https://codecov.io/gh/sunglasses-ai/classy">
        <img alt="Codecov" src="https://codecov.io/gh/sunglasses-ai/classy/branch/main/graph/badge.svg">
    </a>
    <a href="https://optuna.org">
        <img alt="Optuna" src="https://img.shields.io/badge/Optuna-integrated-blue">
    </a>
-->
    <a href="">
        <img alt="Python" src="https://img.shields.io/badge/-Python 3.7--3.9-blue?style=for-the-badge&logo=python&logoColor=white">
    </a>
    <a href="https://pypi.org/project/classy-ml/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/classy-ml?style=for-the-badge&logo=pypi">
    </a>
    <a href="https://pytorch.org/get-started/locally/">
        <img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white">
    </a>
    <a href="https://pytorchlightning.ai/">
        <img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.3+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white">
    </a>
    <a href="https://hydra.cc/">
        <img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.1-89b8cd?style=for-the-badge&labelColor=gray">
    </a>
    <a href="https://black.readthedocs.io/en/stable/">
        <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray">
    </a>
    <br/>
</p>

## Quick Links

- [↗️ Website (FILL LINK)](https://TODO/)
- [🔦 Guide (TODO?)](https://TODO/)
- [💻 Demo (TODO?)](https://TODO/)
- [📓 Documentation (FILL LINKS)](https://TODO/) ( [latest](https://TODO/latest/) | [stable](https://TODO/stable/) | [commit](https://TODO/main/) )
- [✋ Contributing Guidelines (TODO)](CONTRIBUTING.md)
- [⚙️ Continuous Build (TODO)](https://TODO)
- [🌙 Nightly Releases](https://pypi.org/project/classy-ml/#history)


## In this README

- [🚀 Getting Started](#getting-started-using-classy)
- [⚡ Installation](#installation)
    - [Installing via pip](#installing-via-pip)
    - [Installing from source](#installing-from-source)
- [✨ Running Classy](#running-classy)
- [⌨ Commands](#commands)
    - [`classy train`](#classy-train)
    - [`classy predict`](#classy-predict)
    - [`classy evaluate`](#classy-evaluate)
    - [`classy serve`](#classy-serve)
    - [`classy demo`](#classy-demo)
    - [`classy describe`](#classy-describe)
    - [`classy download`](#classy-download)
    - [`classy upload`](#classy-upload)
    - [Enable shell completion](#enabling-shell-completion)
- [🤔 Issues](#issues)
- [❤️ Contributions](#contributions)
- [🤓 Team](#team)


## Getting Started using classy
TODO

## Installation
We strongly recommend using [Conda](https://conda.io/) as the environment manager when dealing with deep learning / data science / machine learning.

`classy` requires Python 3.7 or later, [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://pytorchlightning.ai/).
It's recommended that you install the PyTorch ecosystem **before** installing `classy` by following the instructions on [pytorch.org](https://pytorch.org/).

The preferred way to install `classy` is via `pip`. Just run `pip install classy-ml`.

<!-- `classy` works on *any* platform, as long as it is correctly configured. -->

### Installing via pip

#### Setting up a virtual environment

[Conda](https://conda.io/) can be used set up a virtual environment with the
version of Python required for AllenNLP.  If you already have a Python 3
environment you want to use, you can skip to the 'installing via pip' section.

1.  [Download and install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

2.  Create a Conda environment with Python 3.7-3.9:

    ```yaml
    conda create -n classy python=3.7
    ```

3.  Activate the Conda environment:

    ```yaml
    conda activate classy
    ```

#### Installing the library and dependencies

Simply execute

```yaml
pip install classy-ml
```

and voilà! You're all set.

*Looking for some adventures? Install nightly releases directly from [pypi](https://pypi.org/project/classy-ml/#history)! You will ~~not~~ regret it :)*


### Installing from source
You can also install `classy` by cloning this repository:

```yaml
git clone https://github.com/sunglasses-ai/classy.git
cd classy
```

Follow the steps at [setting up a virtual environment](#setting-up-a-virtual-environment) and then install `classy` by

```yaml
pip install -e .
```

This will make `classy` available in your environment, but using the sources of the cloned repository.

## Running `classy`

## Commands
### `classy train`
### `classy predict`
### `classy evaluate`
### `classy serve`
### `classy demo`
### `classy describe`
### `classy download`
### `classy upload`

### Enabling Shell Completion
To install shell completion, **activate your conda environment** and then execute
```yaml
classy --install-autocomplete
```

From now on, whenever you activate your conda environment with `classy` installed, you are going to have autocompletion when pressing `[TAB]`!

## Team
`classy` is an open-source project developed by [https://sunglasses.ai/](SunglassesAI), a company with the mission to [...]. If you want to know who contributed to this codebase, see our contributors page.



<!--
### Classy Commands

```yaml
classy train (sequence | token | sentence-pair) <dataset-path> [--model-name] [--exp-name] [--device] [--root] [[-c|--config] training.pl_trainer.val_check_interval=1.0 data.pl_module.batch_size=16]
classy predict interactive <model-path> [--device]
classy predict file <model-path> <file-path> [-o|--output-path] [--token-batch-size] [--device]
classy serve <model-path> [-p|--port] [--token-batch-size] [--device]
```

## TODOs

### V0.1
- **luigi**: random -> np.random everywhere
- **luigi**: add inference time to demo
- **edoardo**: create different profiles to train the various tasks (try to build them based on the gpu memory size and report the training time on different gpus)
- **edoardo**: look for and train "una mazzettata" of models
- **niccolò**: classy download
- docs
  - docusaurus
  - comment extensively at least all classes and some important function
  - write readme
- **niccolò**: package install
- **luigi**: Dockerfile

### Later on
- num_workers can't be >1 right now
- pre-commit black (github actions?)
- training on colab notebooks
- logging
- testing


### Classy Commands

```bash
classy train (sequence | token | sentence-pair) <dataset-path> [--model-name] [--exp-name] [--device] [--root] [[-c|--config] training.pl_trainer.val_check_interval=1.0 data.pl_module.batch_size=16]
classy predict interactive <model-path> [--device]
classy predict file <model-path> <file-path> [-o|--output-path] [--token-batch-size] [--device]
classy serve <model-path> [-p|--port] [--token-batch-size] [--device]
```

## TODOs

### V0.1
- **luigi**: random -> np.random everywhere
- **luigi**: add inference time to demo
- **edoardo**: create different profiles to train the various tasks (try to build them based on the gpu memory size and report the training time on different gpus)
- **edoardo**: look for and train "una mazzettata" of models
- **niccolò**: classy download
- docs
  - docusaurus
  - comment extensively at least all classes and some important function
  - write readme
- **niccolò**: package install
- **luigi**: Dockerfile

### Later on
- num_workers can't be >1 right now
- pre-commit black (github actions?)
- training on colab notebooks
- logging
- testing
-->
