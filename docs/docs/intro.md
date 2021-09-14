---
sidebar_position: 1
sidebar_label: Introduction
---

import ReactTermynal from '../src/components/termynal';

# classy

<div style={{textAlign: "center"}}>
  <em>Your data, our code, the hardware you prefer.</em>
  <p></p>
</div>

<div style={{textAlign: "center"}}>
  <a href="https://pypi.org/project/classy-ml" style={{marginRight: ".5rem"}}>
    <img alt="PyPI Package: v0.1" src="https://img.shields.io/badge/PyPI%20Package-v0.1-lightgreen.svg?style=for-the-badge"/>
  </a>
  <a href="https://black.readthedocs.io/en/stable/">
    <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge"/>
  </a>
  <p></p>
</div>

classy is a simple-to-use library for building high-performance Machine Learning models on textual classification tasks.
It wraps the best libraries around ([PyTorch](https://pytorch.org/), [PyTorch Lightning](https://www.pytorchlightning.ai/), [Transformers](https://huggingface.co/transformers/), [Streamlit](https://streamlit.io/)) 
and offers them to users with a simple CLI interface. Using classy does not require any ML knowledge and, on the very opposite, 
its only requirement is that **data** (proprietary datasets, DB dumps, ...) gets organized into straightforward formats. 
Once that is done, classy automatically handles the rest.

Its key features include:
* **simplicity**: very simple-to-use, with no ML knowledge requirement and a usage flow thought around data
* **powerful CLI**: train, present and REST-expose powerful ML models, even with no code whatsoever but a few bash commands
* **fast ML development**: whether for prototyping or production, reduce the time to get things up and running
* **pretrained models**: we offer a number of pre-trained models for different tasks that may suit your needs and speed up the development cycle 
* **modular**: if you have special needs, whether simple (e.g. support a different input format) or advanced (e.g. use a different optimizer),
  classy is extremely modular and offers straightforward hooks to cover every aspect of your desired use case

## Installation

<ReactTermynal>
  <span data-ty="input">pip install classy-ml</span>
  <span data-ty="progress"></span>
  <span data-ty>Successfully installed classy-ml</span>
</ReactTermynal>

## Example Walkthrough

You have the following data at disposal:

| User | Product-ID | Review | Overall Judgement |
| ----------- | ----------- | ----------- | ----------- |
| U1 | P1 | I really like this headphones | positive | 
| U2 | P1 | Sound is terrible! I am returning this headphones right away. | negative |
| ... | ... | ... | ... |

That is, a list of reviews made by users for a given product, along with their overall judgement. Now you want to train
a classification model on this data so that, given a new review as input, it would yield whether its overall judgement is
positive or negative.

:::info

This is a *Sequence Classification* problem. It is one of the three major formulations of text classification tasks,
along with *Sentence-Pair Classification* and *Token Classification*. All of these are supported by classy.

:::

The following steps depict how you would do it with classy:

**Step 1**: Organize your data into a *.tsv* file

```python
def dump_data_on_tsv(corpus, output_file):
    # todo implement here
    pass

corpus, output_file = load_corpus(), 'data/output.tsv'
dump_data_on_tsv(corpus, output_file)
```

This is the only part where some coding is required. However, note that it need not be Python code 
(if you are an *AWK* fan, feel free to use it).

**Step 2**: Train a model

<ReactTermynal>
  <span data-ty="input">classy train sequence data/output.tsv</span>
  <span data-ty="progress"></span>
  <span data-ty>Training completed</span>
</ReactTermynal>

<p />

:::info

*sequence* in the above command tells classy to train a *Sequence Classification* model. This is the only thing, beside 
organizing data, that classy expects you to do.

:::

**Step 3.a**: Present a demo of a model

<ReactTermynal>
  <span data-ty="input">classy demo experiments/sequence-bert/2021-09-08/12-11-57/checkpoints/best.ckpt</span>
  <span data-ty startDelay="2000">Demo up and running at http://0.0.0.0:8000</span>
</ReactTermynal>

<p />

Now you can check out out the demo!

![Classy Demo](/img/intro/demo.png)

**Step 3.b**: Expose via REST API

<ReactTermynal>
  <span data-ty="input">classy serve experiments/sequence-bert/2021-09-08/12-11-57/checkpoints/best.ckpt</span>
  <span data-ty startDelay="2000">REST API up and running at http://0.0.0.0:8000</span>
  <span data-ty>Checkout the OpenAPI docs at http://0.0.0.0:8000/docs</span>
  <span data-ty="input">curl -X 'POST' 'http://localhost:8000/' -H 'accept: application/json' -H 'Content-Type: application/json' -d '[{'{'}"sequence": "I wish I had never bought these terrible headphones!"{'}'}]'</span>
  <span data-ty startDelay="2000">[{'{'}"sequence":"I wish I had never bought these terrible headphones!","label":"0"{'}'}]</span>
</ReactTermynal>

<p />

We also automatically generate the OpenAPI documentation page!

![Classy Serve Docs](/img/intro/serve-docs.png)

**Step 3.c**: Bash-interactive predict

<ReactTermynal>
  <span data-ty="input">classy predict interactive experiments/sequence-bert/2021-09-08/12-11-57/checkpoints/best.ckpt</span>
  <span data-ty="input" data-ty-prompt="Enter source text: ">I wish I had never bought these terrible headphones!</span>
  <span data-ty startDelay="2000">  # prediction: negative</span>
  <span data-ty data-ty-prompt="Enter source text: "></span>
</ReactTermynal>

<p />

**Step 3.d**: Predict every review stored in a target file

<ReactTermynal>
  <span data-ty="input">cat target.tsv | head -1</span>
  <span data-ty>I wish I had never bought these terrible headphones!</span>
  <span data-ty="input">classy predict file experiments/sequence-bert/2021-09-08/12-11-57/checkpoints/best.ckpt target.tsv -o target.out.tsv</span>
  <span data-ty="progress"></span>
  <span data-ty>Prediction complete</span>
  <span data-ty="input">cat target.out.tsv | head -1</span>
  <span data-ty>I wish I had never bought these terrible headphones!    negative</span>
</ReactTermynal>

<p />

