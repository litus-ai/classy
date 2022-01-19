---
title: Introduction
sidebar_position: 1
sidebar_label: Introduction
---

import ReactTermynal from '/src/components/termynal';

`classy` is a simple-to-use library for building high-performance Machine Learning models in NLP.
It wraps the best libraries around ([PyTorch](https://pytorch.org/), [PyTorch Lightning](https://www.pytorchlightning.ai/),
[Transformers](https://huggingface.co/transformers/), [Streamlit](https://streamlit.io/), ...)
and offers them to users with a simple CLI interface.

## Installation

<ReactTermynal>
  <span data-ty="input">pip install classy-core</span>
  <span data-ty="progress"></span>
  <span data-ty>Successfully installed classy-core</span>
</ReactTermynal>

## Example Walkthrough

You have the following data at disposal:

| User | Product-ID | Review | Overall Judgement |
| ----------- | ----------- | ----------- | ----------- |
| U1 | P1 | I really like this headphones | positive |
| U2 | P1 | Sound is terrible! I am returning these headphones right away. | negative |
| ... | ... | ... | ... |

That is, a list of reviews made by users for a given product, along with their overall judgement. Now, say you want to train
a classification model on this data so that, given a new review as input, it would yield whether its overall judgement is
positive or negative.

:::info

This is a **Sequence Classification** problem, specifically *Sentiment Classification*.

:::

In a nutshell, this is how you would do it with `classy`:

### Data Organization

By default, `classy` is able to work with `.tsv` and `.jsonl` files. In the case of `.tsv` for **Sequence Classification**,
`classy` needs the file to have only two columns, one with the sequence and the other with the label.

Say your review data is already stored in a `.tsv` file (with four columns, as we saw above), all you need to do is:

```bash
$ cut -f3,4 data/raw.tsv > data/classy-sentiment.tsv
```

:::tip

`classy` supports many [tasks and input formats](/docs/reference-manual/tasks-and-formats), and is independent of how you
convert your data to a `classy`-compatible format (i.e., `sed`, `awk`, and a `python` script are all viable options).

:::

### Training
Now that our data is `classy`-compatible, we can train our model! Let us call it `sequence-example`:

<ReactTermynal>
  <span data-ty="input">classy train sequence data/classy-sentiment.tsv -n sequence-example</span>
  <span data-ty="progress"></span>
  <span data-ty>Training completed</span>
</ReactTermynal>

<p />

:::info

`sequence` in the above command tells classy to train a **Sequence Classification** model. This is the only thing, beside
organizing data, that classy expects you to do. We'll go back to this later on.

:::

### Using your trained model

Now that we have trained our model, `classy` offers a number of commands that let you use it:

- `classy demo` spawns a [Streamlit](https://streamlit.io/)-based interface that lets you query your model visually;
- `classy predict` lets you choose whether to use your model for file prediction or in bash-interactive mode (similarly to `demo`, but in the terminal);
- `classy serve` spawns a REST API (through [FastAPI](https://fastapi.tiangolo.com/)) that you can query using any REST-compatible client (`requests`, `curl`, you name it);
- `classy evaluate` runs your model on a given test set and reports metrics on the task.
