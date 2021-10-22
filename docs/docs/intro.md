---
title: Introduction
sidebar_position: 1
sidebar_label: Introduction
---

import ReactTermynal from '../src/components/termynal';

classy is a simple-to-use library for building high-performance Machine Learning models in NLP.
It wraps the best libraries around ([PyTorch](https://pytorch.org/), [PyTorch Lightning](https://www.pytorchlightning.ai/), [Transformers](https://huggingface.co/transformers/), [Streamlit](https://streamlit.io/), ...) 
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
| U2 | P1 | Sound is terrible! I am returning this headphones right away. | negative |
| ... | ... | ... | ... |

That is, a list of reviews made by users for a given product, along with their overall judgement. Now you want to train
a classification model on this data so that, given a new review as input, it would yield whether its overall judgement is
positive or negative.

:::info

This is a *Sequence Classification* problem.

:::

The following steps depict how you would do it with classy:

### Data Organization

Organize your data into a *.tsv* file:

```python
def dump_data_on_tsv(corpus, output_file):
    # todo implement here
    pass

corpus, output_file = load_corpus(), 'data/output.tsv'
dump_data_on_tsv(corpus, output_file)
```

This is the only part where some coding is required. 

:::note 

You don't have to use Python here, you can use any tool as far as data gets organized into a .tsv file (if you are an *AWK* fan, for it).

:::

### Training

Train a model, named *sequence-example*:

<ReactTermynal>
  <span data-ty="input">classy train sequence data/output.tsv -n sequence-example</span>
  <span data-ty="progress"></span>
  <span data-ty>Training completed</span>
</ReactTermynal>

<p />

:::info

*sequence* in the above command tells classy to train a *Sequence Classification* model. This is the only thing, beside 
organizing data, that classy expects you to do. We'll go back to this later on.

:::

### Presenting

Present a demo of *sequence-example*:

<ReactTermynal>
  <span data-ty="input">classy demo sequence-example</span>
  <span data-ty data-ty-start-delay="2000">Demo up and running at http://0.0.0.0:8000</span>
</ReactTermynal>

<p />

Check out out the demo at http://0.0.0.0:8000!

![Classy Demo](/img/intro/demo.png)

### Exposing via REST API

Expose *sequence-example* via REST API:

<ReactTermynal>
  <span data-ty="input">classy serve sequence-example</span>
  <span data-ty data-ty-start-delay="2000">REST API up and running at http://0.0.0.0:8000</span>
  <span data-ty>Checkout the OpenAPI docs at http://0.0.0.0:8000/docs</span>
  <span data-ty="input">curl -X 'POST' 'http://localhost:8000/' -H 'accept: application/json' -H 'Content-Type: application/json' -d '[{'{'}"sequence": "I wish I had never bought these terrible headphones!"{'}'}]'</span>
  <span data-ty data-ty-start-delay="2000">[{'{'}"sequence":"I wish I had never bought these terrible headphones!","label":"0"{'}'}]</span>
</ReactTermynal>

<p />

We also automatically generate the OpenAPI documentation page!

![Classy Serve Docs](/img/intro/serve-docs.png)


### Predicting

Use *sequence-example* to assign a label to every review stored in a target file:

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

