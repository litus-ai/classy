---
sidebar_position: 2
title: Installation
---

:::tip
We strongly recommend using [Conda](https://conda.io/) as the environment manager when dealing with deep learning / data science / machine learning.
:::

`classy` requires Python 3.8 or later, and is built on [PyTorch Lightning](https://pytorchlightning.ai/).
It's recommended that you install the PyTorch ecosystem **before** installing `classy` by following the instructions on [pytorch.org](https://pytorch.org/).

Or, simply put:
```yaml
conda install pytorch cudatoolkit=CUDA_VERSION -c pytorch
```

:::tip
Don't know what `CUDA_VERSION` you have?
[Check this link](https://stackoverflow.com/a/68499241/1908499).
:::

The preferred way to install `classy` is via `pip`. Just run `pip install classy-core`.

<!-- `classy` works on *any* platform, as long as it is correctly configured. -->

## Installing via pip

#### Setting up a virtual environment

[Conda](https://conda.io/) can be used set up a virtual environment with the
version of Python required for `classy`.  If you already have a Python 3
environment you want to use, you can skip to the
[Installing the library and dependencies](/docs/installation/#installing-the-library-and-dependencies) section.

1.  [Download and install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

2.  Create a Conda environment with Python 3.8+:

    ```yaml
    conda create -n classy python=3.8
    ```

3.  Activate the Conda environment:

    ```yaml
    conda activate classy
    ```

#### Installing the library and dependencies

Simply execute

```yaml
pip install classy-core
```

and voilà! You're all set.

*Looking for some adventures? Install nightly releases directly from [pypi](https://pypi.org/project/classy-core/#history)! You will ~~not~~ regret it :)*


## Installing from source
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

## Using classy via docker

Alternatively, we also release and mantain a Docker image with everything already set up. In order to use it locally,
you need to install:
* [docker](https://docs.docker.com/get-docker/)
* [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Once set up, you can just run `docker run --gpus all -it poccio/classy bash` (or checkout [poccio/classy](https://hub.docker.com/repository/docker/poccio/classy)
for specific version tags) and you'll get a shell on a fully-setup container, with a *classy* conda environment already
created.

Alternatively, if you use external cloud computing platforms that support docker images (e.g. [vast.ai](https://vast.ai/)),
just providing the image name (`poccio/classy`) should suffice (it's hosted on Dockerhub).
