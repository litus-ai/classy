---
sidebar_position: 3
title: CLI
---

import ReactTermynal from '../../../src/components/termynal';

Once task identification and data organization are complete, your "have-to-code" section is over, and you can use classy
CLI to handle the rest, covering every step during the lifecycle of a ML model:
* train
* predict (both file-based and bash-interactive)
* evaluate
* serve via a REST API
* present a demo

## Train

The core syntax of the train command is the following:
```bash
classy train 
    <task>                # any in *{sequence,token,sentence-pair}*
    <path-to-dataset>     # path to your dataset
    -n <exp-name>         # name you want to give to your model
    -d <device>           # device on which to train ("cpu" for cpu, or device number for gpu)
```

:::info

If a device is not provided explicitly, classy will search and use a gpu if present, resorting otherwise to cpu.

:::

Your model and experiment data will be saved in *experiments/*&lt;exp-name&gt;*/current-day/current-time/*.

:::tip

If you want to transfer your model to a different pc, the simplest way is to transfer the entire 
*experiments/*&lt;exp-name&gt;*/current-day/current-time/* folder. However, multiple checkpoints, i.e. the model at different
moments in the training, might be present in the *checkpoints/* folder; to speed up the transfer, you might want to
consider moving only one of them, for instance the best one, *checkpoints/best.ckpt*.

:::

Note that *&lt;path-to-dataset&gt;* is a bit of a special parameter and can be either:
* **folder**, or, actually, a *ML-ready* folder: that is, it must contain a training file, named *train.#* 
  (# denotes a classy supported extension), and, optionally, a validation file and a test file, *validation.#* and *test.#*
* **file**: classy uses the file provided to automatically generate an ML-ready folder for you, storing it in the
  *data/* folder inside the current experiment
  
:::info

In the folder case, if *validation.#* is not present, classy automatically generates it by reserving some samples
from the training file: the new training and validation files are saved in the *data/* folder inside the current experiment.
This does not hold for *test.#* and, if not present, classy **will not** create it.

:::

<ReactTermynal>
  <span data-ty="input">classy train sequence data/output.tsv -n sequence-example</span>
  <span data-ty="progress"></span>
  <span data-ty>Training completed</span>
  <span data-ty="input">ls experiments</span>
  <span data-ty>sequence-sample/</span>
  <span data-ty="input">tree -L 2 sequence-example</span>
  <span className="data-ty-treefix" data-ty>
    <div>
        {`experiments/sequence-example/<day>/<time>/
          ├── checkpoints
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

For all CLI commands that involve using a trained model, you can use 3 ways to specify it:
```bash
# the <exp-name> you used at training time (classy will find and use the latest best.ckpt)
classy <cmd> sequence-example ... 
# path to model folder
classy <cmd> experiments/sequence-example/ ... 
# path to specific model experiment <day>/<time> 
classy <cmd> experiments/sequence-example/<day>/<time> ... 
# path to checkpoint inside a model experiment folder
classy <cmd> experiments/sequence-example/<day>/<time>/checkpoints/epoch=00-val_loss=0.51.ckpt
```

:::tip

For all CLI commands, you can checkout `classy <command> -h` for more information and parameters.

:::

## Predict

You can use `classy predict` to perform predictions with a trained model, using 

Two modes are supported:
* **file-based prediction** 
* **bash-interactive**

File-based prediction allows you to automatically tag files. Such files can be in any supported format and need not contain 
label information: that is, the corresponding area, such as the second column for .tsv files in sequence classification, 
can be missing (if present, it will just be ignored).

<ReactTermynal>
  <span data-ty="input">cat target.tsv | head -1</span>
  <span data-ty>I wish I had never bought these terrible headphones!</span>
  <span data-ty="input">classy predict file sequence-example target.tsv -o target.out.tsv</span>
  <span data-ty="progress"></span>
  <span data-ty>Prediction complete</span>
  <span data-ty="input">cat target.out.tsv | head -1</span>
  <span data-ty>I wish I had never bought these terrible headphones!    negative</span>
</ReactTermynal>

<p />

On the other hand, bash-interactive predictions allows you to interactively query models via bash. 
Here, the input is expected to be a tab-separated string that is interpreted as the input in .tsv files.

<ReactTermynal>
  <span data-ty="input">classy predict interactive sequence-sample</span>
  <span data-ty="input" data-ty-prompt="Enter source text: ">I wish I had never bought these terrible headphones!</span>
  <span data-ty data-ty-start-delay="2000">  # prediction: negative</span>
  <span data-ty data-ty-prompt="Enter source text: "></span>
</ReactTermynal>

<p />

## Evaluate

You can use `classy evaluate` to evaluate your trained model against a dataset. If no dataset is explicitly
provided, classy will try to locate the test set provided to `classy train` (if any).

<ReactTermynal>
  <span data-ty="input">classy evaluate sequence-bert</span>
  <span data-ty="progress"></span>
  <span data-ty># accuracy: 0.9</span>
  <span data-ty># classification metrics:</span>
  <span data-ty>...</span>
  <span data-ty>    # f1: 0.9</span>
  <span data-ty>...</span>
</ReactTermynal>

<p />

:::caution

If you move your model to a different pc, automatically inferring the location of the test set will fail 
unless it was also moved (and placed in a symmetric location in the file-system). Should it fail, provide
explicitly its path in this case.

:::

## Serve

You can use `classy serve` to expose your model via a REST API with [FastAPI](https://fastapi.tiangolo.com/).

<ReactTermynal>
  <span data-ty="input">classy serve sequence-example</span>
  <span data-ty data-ty-start-delay="2000">REST API up and running at http://0.0.0.0:8000</span>
  <span data-ty>Checkout the OpenAPI docs at http://0.0.0.0:8000/docs</span>
  <span data-ty="input">curl -X 'POST' 'http://localhost:8000/' -H 'accept: application/json' -H 'Content-Type: application/json' -d '[{'{'}"sequence": "I wish I had never bought these terrible headphones!"{'}'}]'</span>
  <span data-ty data-ty-start-delay="2000">[{'{'}"sequence":"I wish I had never bought these terrible headphones!","label":"0"{'}'}]</span>
</ReactTermynal>

<p />

You can also checkout the OpenAPI documentation page we automatically generate at *http://0.0.0.0:8000/docs*.

![Classy Serve Docs](/img/intro/serve-docs.png)

:::tip

By default, `classy serve` uses port 8000. Use the *-p* parameter to specify a different one.

:::

## Demo

You can use `classy demo` to spawn a [Streamlit](https://streamlit.io/) demo of your model.

<ReactTermynal>
  <span data-ty="input">classy demo sequence-example</span>
  <span data-ty data-ty-start-delay="2000">Demo up and running at http://0.0.0.0:8000</span>
</ReactTermynal>

<p />

Now you can check out out the demo at *http://0.0.0.0:8000*!

![Classy Demo](/img/intro/demo.png)

## Additional Commands

Besides these commands, classy CLI also offers you:
* `classy describe`, a data-analysis tool with general and task-specific stats over your dataset
* `classy download`, to download pre-trained models 
* `classy upload`, to upload your trained models 

### Describe

TODO

### Upload

You can use `classy upload` to save a trained model on the [HuggingFace Model Hub](https://huggingface.co/models). 

:::note
Uploading to the HF Hub **requires** a HuggingFace account. 

You can create one [here](https://huggingface.co/join) and then run
`huggingface-cli login` to perform the login from your machine.
:::

<ReactTermynal>
  <span data-ty="input">classy train [...] -n modelname</span>
  <span data-ty="input">classy upload modelname</span>
  <span data-ty="progress"></span>
  <span data-ty>Model uploaded to the HuggingFace Hub.</span>
  <span data-ty>You can download it running `classy download username@modelname`!</span>
</ReactTermynal>

### Download

You can use `classy download` to download a model previously uploaded by someone else with `classy upload`.

:::caution
`classy download` currently **only** works with models trained with the base `classy` library. 

Support for user-defined models is being discussed internally. 
:::

<ReactTermynal>
  <span data-ty="input">classy download username@modelname</span>
  <span data-ty="input">classy serve username@modelname</span>
  <span data-ty data-ty-start-delay="2000">Demo up and running at http://0.0.0.0:8000</span>
</ReactTermynal>

<br />

:::info
Models downloaded through `classy download` are stored under `~/.cache/sunglasses-ai/classy`.
:::

:::tip
Once a model has been downloaded, it is accessible from anywhere on the machine using any of the model-based `classy` commands (e.g. `serve`, `predict`, etc).
:::
