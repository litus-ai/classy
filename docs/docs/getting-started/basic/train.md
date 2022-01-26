---
sidebar_position: 4
title: Training your model
---

import ReactTermynal from '/src/components/termynal';

We are ready to train our first model with `classy`!
Since we are doing *Named Entity Recognition*, and we want to go with a fast model (i.e., `distilbert`),
let's name this experiment *fast-ner*:

<ReactTermynal>
  <span data-ty="input">classy train token data/train.tsv -n fast-ner --profile distilbert</span>
  <span data-ty="progress"></span>
  <span data-ty>Training completed</span>
</ReactTermynal>

<p />


:::tip

Your model and experiment data will be saved in `experiments/<exp-name>/YYYY-MM-DD/HH-mm-ss/`.

`classy` automatically saves *best* and *last* checkpoints, as well as pre- and post-trainer initialization configurations.

:::

:::info

*token* in the above command tells classy to train a *Token Classification* model. This is the only thing, besides
organizing data, that classy expects you to do. We'll go back to this later on.

:::
