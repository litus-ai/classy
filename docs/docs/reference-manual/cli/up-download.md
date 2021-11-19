---
sidebar_position: 4
title: upload/download
---

import ReactTermynal from '/src/components/termynal';

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
