---
sidebar_position: 1
title: Overall Structure
---

In `classy`, we specify every detail of training and model configurations through nice `.yaml` files,
using the amazing [Hydra](https://hydra.cc/) library.

:::info

While you can read this section and work with classy config structure, without knowing Hydra, we recommend
going through its [tutorial](https://hydra.cc/docs/tutorials/intro) before proceeding.

:::

This allows you to change and swap parts easily. For instance, imagine you have been fine-tuning BERT on some Token
Classification task, with this configuration:

```yaml
<...>
transformer_model: bert-large-cased
use_last_n_layers: 1
fine_tune: True
optim_conf:
  <...>
```

If you wanted to give a try keeping BERT weights frozen, you can just go with:

```yaml
_target_: 'classy.pl_modules.hf.HFTokensPLModule'
transformer_model: bert-large-cased
use_last_n_layers: 1
fine_tune: False
optim_conf:
  <...>
```

:::tip

These `.yaml` files also help you to **track** what changes you made on some experiment as the configuration is saved alongside
the model in the experiment folder.

:::

## A Minimal Example

However, if you were to specify every aspect of your experiment in a single `.yaml` file, it would eventually become gigantic
and pretty much useless. To avoid this, we use Hydra *config groups*, which is just a fancy name to say that the
experiment details are grouped by functionality and stored inside dedicated folders. That is:

```bash
$ tree -L 1 configurations/
configurations/
├── data/
│   └── token.yaml
├── model
│   └── token.yaml
├── training
│   └── token.yaml
└── root.yaml
```

`data/token.yaml` defines data-related configurations for your **Token Classification** experiment, while `model/token.yaml` and
`training/token.yaml` specify model (e.g. architecture) and training (e.g. gradient accumulation) aspects.

`root.yaml` is the yaml *orchestrator*, that is, it defines global variables and specify which yaml file in each folder
should be used:

```yaml title=root.yaml
# global variables
task: token
project_name: classy
<...>

# here specify yaml file to use for each config group
# syntax: the name of the file (without .yaml extension) contained in the corresponding folder
defaults:
  - data: token
  - model: token
  - logging: default
  - _self_  # this is some hydra-specific machinery (you can ignore it, but leave it at the end of the defaults list)
```

## Full Structure

As a matter of fact, there are quite a few details more `data/`, `model/` and `train/` that you would want to specify.
Thus, the actual structure of the config groups is the following:

```bash
$ tree -L 1 configurations/
configurations/
├── callbacks/          # used to define callbacks that are executed during training (at precise steps, like every end of validation)
├── data/               # data-related configuration (e.g. PyTorch Dataset)
├── model/              # model-related configuration (e.g. architecture)
├── prediction/         # data-configuration to be used at prediction time
└── training/           # training configuration
```

Besides, there are 5 root `.yaml` files already defined, one for each task:
* `qa.yaml`
* `sentence-pair.yaml`
* `sequence.yaml`
* `token.yaml`
* `generation.yaml`

:::info

If you manually inspect the `configurations/` folder, you'll see there are 3 additional folders: `logging/` and
`profiles/`. You can safely ignore the first two, and we cover `profiles/` in depth [here](/docs/getting-started/customizing-things/changing-profile).

:::
