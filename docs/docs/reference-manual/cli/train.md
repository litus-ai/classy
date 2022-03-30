---
sidebar_position: 1
title: train
---

import ReactTermynal from '/src/components/termynal';

The core syntax of the train command is the following:
```bash
classy train
    <task>                # any in {sequence,token,sentence-pair}
    <path-to-dataset>     # path to your dataset
    -n <exp-name>         # name you want to give to your model
    -d <device>           # device on which to train ("cpu" for cpu, or device number for gpu)
    [--print]
```

:::info

If a device is not provided explicitly, `classy` will search and use a gpu if present, resorting otherwise to cpu.

:::

Your model and experiment data will be saved in *experiments/*&lt;exp-name&gt;*/current-day/current-time/*.

:::tip

If you want to transfer your model to a different pc, the simplest way is to transfer the entire
*experiments/*&lt;exp-name&gt;*/current-day/current-time/* folder. However, multiple checkpoints, i.e. the model at different
moments in the training, might be present in the *checkpoints/* folder; to speed up the transfer, you might want to
consider moving only one of them, for instance the best one, *checkpoints/best.ckpt*.

:::

**Note** that *&lt;path-to-dataset&gt;* is a bit of a special parameter and can be either:
* a **folder**, or, actually, a *ML-ready* folder: that is, it must contain a training file, named *train.#*
  (# denotes a classy supported extension), and, optionally, a validation file and a test file, *validation.#* and *test.#*
* a **file**: classy uses the file provided to automatically generate an ML-ready folder for you, storing it in the
  *data/* folder inside the current experiment
* a **yaml file**: classy reads the yaml file and infer the datasets involved in the training process from it. It should be organized as follows
```yaml
# Optional parameter that tells classy which is your favourite format.
# If classy has to save something, it will save it in the format you specify, 
#  using the corresponding DataDriver.
main_data_driver: json

# You can specify a split using a dictionary path -> extension
#  that tells classy the dataset involved and which DataDriver 
#  to use for each dataset.
train_dataset:
  "first-train-dataset-path.tsv": "tsv"
  "second-train-dataset-path.json": "my-special-format"

# You can also specify a split as a list of paths, and the DataDriver
#  to use for each dataset will be inferred from the extension.
validation_dataset:
  - "fist-validation-dataset.tsv"
  - "second-validation-dataset.tsv"

# And finally, you can specify a split as single path
test_dataset: "test-dataset.json"
```
:::info

In both _folder_ and _yaml file_ cases, if the validation dataset is not specified, classy automatically generates it by reserving some samples
from the training file: the new training and validation files are saved in the *data/* folder inside the current experiment.
This does not hold for the test set and, if not present, classy **will not** create it.

:::

<ReactTermynal>
  <span data-ty="input">classy train sequence data/sequence-dataset -n sequence-example</span>
  <span data-ty="progress"></span>
  <span data-ty>Training completed</span>
  <span data-ty="input">ls experiments</span>
  <span data-ty>sequence-sample/</span>
  <span data-ty="input">tree -L 2 sequence-example</span>
  <span className="data-ty-treefix" data-ty>experiments/sequence-example/&lt;day&gt;/&lt;time&gt;/
    <div>
        {`├── checkpoints
            │   ├── best.ckpt
            │   ├── epoch=00-val_loss=0.51.ckpt
            │   └── last.ckpt
            ├── data
            │   ├── train.tsv
            │   └── validation.tsv
            └── ...
        `.split('\n').map( (it, i) => <p key={i} style={{lineHeight: "1.0"}}>{it}</p>)}
    </div>
  </span>
</ReactTermynal>

<p />

For all CLI commands that involve using a trained model, you can use 4 ways to specify it:
```bash
# the <exp-name> you used at training time (classy will search in the experiments/ folder and use the latest best.ckpt)
classy <cmd> sequence-example ...
# path to model folder
classy <cmd> experiments/sequence-example/ ...
# path to specific model experiment <day>/<time>
classy <cmd> experiments/sequence-example/<day>/<time> ...
# path to checkpoint inside a model experiment folder
classy <cmd> experiments/sequence-example/<day>/<time>/checkpoints/epoch=00-val_loss=0.51.ckpt
```

:::tip

For all CLI commands, you can execute `classy <command> -h` for additional information and parameters.

:::


## Visualizing your Configuration

`classy train` has an option that lets you visualize the full materialized configuration in your terminal, showing you
where each component comes from (including overridden configuration groups, overrides, profiles, etc.).
To print it, simply append `--print` to your current `classy train` command!

![Classy Train Print - Token](/img/intro/classy-train-print-tok.png)
