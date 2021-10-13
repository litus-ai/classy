<div align="center">
    <br>
    <img alt="classy logo" src="img/logo.png" width="400"/>
    <p>
    A PyTorch-based library for fast prototyping and sharing of deep neural network models. 
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
    <a href="https://optuna.org">
        <img alt="Optuna" src="https://img.shields.io/badge/Optuna-integrated-blue">
    </a>
-->
    <a href="">
        <img alt="Python" src="https://img.shields.io/badge/Python 3.7--3.9-blue?style=for-the-badge&logo=python&logoColor=white">
    </a>
    <a href="https://pypi.org/project/classy-ml/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/classy-ml?style=for-the-badge&logo=pypi">
    </a>
    <a href="https://pytorch.org/get-started/locally/">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch 1.8+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white">
    </a>
    <a href="https://pytorchlightning.ai/">
        <img alt="Lightning" src="https://img.shields.io/badge/Lightning 1.4.5-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white">
    </a>
    <a href="https://hydra.cc/">
        <img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.1.1-89b8cd?style=for-the-badge&labelColor=gray">
    </a>
    <a href="https://black.readthedocs.io/en/stable/">
        <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray">
    </a>
    <a href="https://codecov.io/gh/sunglasses-ai/classy">
        <img alt="Codecov" src="https://img.shields.io/codecov/c/github/sunglasses-ai/classy/main?label=cov&logo=codecov&style=for-the-badge&token=S0PMBCTG73">
    </a>
    <br/>
</p>

## Quick Links

<!-- - [💻 Demo (TODO?)](https://TODO/) 
- [⚙️ Continuous Build (TODO)](https://TODO)-->
- [↗️ Website](http://151.100.179.12:30000/)
- [🔦 Guide](http://151.100.179.12:30000/docs/intro)
- [💻 Template and Examples](https://github.com/sunglasses-ai/classy-template)
- [📓 Documentation (FILL LINKS)](https://TODO/) ( [latest](https://TODO/latest/) | [stable](https://TODO/stable/) | [commit](https://TODO/main/) )
- [✋ Contributing Guidelines (TODO)](CONTRIBUTING.md)
- [🌙 Nightly Releases](https://pypi.org/project/classy-ml/#history)


## In this README

- [🚀 Getting Started](#getting-started-using-classy)
- [⚡ Installation](#installation)
- [⌨ Running Classy](#running-classy)
    - [`classy train`](#classy-train)
    - [`classy predict`](#classy-predict)
    - [`classy evaluate`](#classy-evaluate)
    - [`classy serve`](#classy-serve)
    - [`classy demo`](#classy-demo)
    - [`classy describe`](#classy-describe)
    - [`classy upload`](#classy-upload)
    - [`classy download`](#classy-download)
    - [Enable `classy` shell completion](#enabling-shell-completion)
- [🤔 Issues](#issues)
- [❤️ Contributions](#contributions)
- [🤓 Team](#team)


## Getting Started using classy
If this is your first time meeting `classy`, don't worry! We have plenty of resources to help you learn how it works and what it can do for you.

For starters, have a look at our [amazing website](http://151.100.179.12:30000) and [our documentation](http://151.100.179.12:30000/docs/intro)!

If you want to get your hands dirty right away, have a look at our [base classy template](https://github.com/sunglasses-ai/classy-template). There are a few example branches there that you can look at to get to know `classy`!

## Installation

*For a more in-depth installation guide (covering also installing from source and through docker), please visit our [installation page](http://151.100.179.14:30000/docs/getting-started/installation).*

If you are using one of our [templates](https://github.com/sunglasses-ai/classy-template), there is a handy `setup.sh` script you can use that will execute the commands to create the environment and install `classy` for you.

### Installing via pip

#### Setting up a virtual environment

We strongly recommend using [Conda](https://conda.io/) as the environment manager when dealing with deep learning / data science / machine learning. It's also recommended that you install the PyTorch ecosystem **before** installing `classy` by following the instructions on [pytorch.org](https://pytorch.org/)

If you already have a Python 3 environment you want to use, you can skip to the [installing via pip](#installing-via-pip) section.

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


## Running `classy`
Once it is installed, `classy` is available as a command line tool. It offers a wide variety of subcommands, all listed below. Detailed guides and references for each command is available [in the documentation](http://TODO). Every one of `classy`'s subcommands have a `-h|--help` flag available which details the various arguments & options you can use (e.g., `classy train -h`).

### `classy train`
In its simplest form, `classy train` lets you train a transformer-based neural network for one of the tasks supported by `classy` (see TODO).

```yaml
classy train sentence-pair path/to/dataset/folder-or-file -n my-model
```
The command above will train a model to predict a label given a pair of sentences as input (e.g., Natural Language Inference or NLI) and save it under `experiments/my-model`. This same model can be further used by all other `classy` commands which require a `classy` model (`predict`, `evaluate`, `serve`, `demo`, `upload`).

### `classy predict`
`classy predict` actually has two subcommands: `interactive` and `file`. 

The first loads the model in memory and lets you try it out through the shell directly, so that you can test the model you trained and see what it predicts given some input. It is particularly useful when your machine cannot open a port for [`classy demo`](#classy-demo).

The second, instead, works on a file and produces an output where, for each input, it associates the corresponding predicted label. It is very useful when doing pre-processing or when you need to evaluate your model (although we offer [`classy evaluate`](#classy-evaluate) for that).

### `classy evaluate`
`classy evaluate` lets you evaluate your model on standard metrics for the task your model was trained upon. Simply run `classy evaluate my-model path/to/file -o path/to/output/file` and it will dump the evaluation at `path/to/output/file`

### `classy serve`
`classy serve <model>` loads the model in memory and spawns a REST API you can use to query your model with any REST client.

### `classy demo`
`classy demo <model>` spawns a [Streamlit](https://streamlit.io) interface which lets you quickly show and query your model.

### `classy describe`
`classy describe <task> --dataset path/to/dataset` runs some common metrics on a file formatted for the specific task. Great tool to run **before** training your model!

### `classy upload`
`classy upload <model>` lets you upload your `classy`-trained model on the [HuggingFace Hub](https://huggingface.co) and lets other users download / use it. (NOTE: you need a HuggingFace Hub account in order to upload to their hub)

Models uploaded via `classy upload` will be available for download by other classy users by simply executing `classy download username@model`.

### `classy download`
`classy download <model>` downloads a previously uploaded `classy`-trained model from the [HuggingFace Hub](https://huggingface.co) and stores it on your machine so that it is usable with any other `classy` command which requires a trained model (`predict`, `evaluate`, `serve`, `demo`, `upload`). 

You can find [SunglassesAI](http://sunglasses.ai)'s list of pre-trained models [here](https://huggingface.co/sunglasses-ai). 

Models uploaded via `classy upload` are available by doing `classy download username@model`.

### Enabling Shell Completion
To install shell completion, **activate your conda environment** and then execute
```yaml
classy --install-autocomplete
```

From now on, whenever you activate your conda environment with `classy` installed, you are going to have autocompletion when pressing `[TAB]`!

## Issues
You are more than welcome to file issues with either feature requests, bug reports, or general questions. If you already found a solution to your problem, don't hesitate to share it. Suggestions for new best practices and tricks are always welcome!

## Contributions
We warmly welcome contributions from the community. If it is your first time as a contributor, we recommend you start by reading our CONTRIBUTING.md guide.

Small contributions can be made directly in a pull request. For contributing major features, we recommend you first create a issue proposing a design, so that it can be discussed before you risk wasting time.

Pull requests (PRs) must have one approving review and no requested changes before they are merged. 
As `classy` is primarily driven by SunglassesAI, we reserve the right to reject or revert contributions that we don't think are good additions or might not fit into our roadmap.


## Team
`classy` is an open-source project developed by [SunglassesAI](https://sunglasses.ai/), a company with the mission to [...]. If you want to know who contributed to this codebase, see our contributors page.



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
