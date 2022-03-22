---
sidebar_position: 5
title: import/export
---

import ReactTermynal from '/src/components/termynal';

### Export

You can use `classy export` to save a zip file of a trained model to be shared with others.


<ReactTermynal>
  <span data-ty="input">classy train [...] -n modelname</span>
  <span data-ty="input">classy export modelname</span>
  <span data-ty="progress"></span>
  <span data-ty>Model saved to classy-export-modelname.zip</span>
</ReactTermynal>

### Import

You can use `classy import` to import a model previously exported by someone else with `classy export`. By default,
everything will be imported under your `experiments/` folder (if you run from a classy project),
but you can specify a target path via the `--exp-dir` argument.


<ReactTermynal>
  <span data-ty="input">classy import classy-export-modelname.zip</span>
  <span data-ty="input">classy serve modelname</span>
  <span data-ty data-ty-start-delay="2000">Demo up and running at http://0.0.0.0:8000</span>
</ReactTermynal>

<br />

:::tip
Once a model has been imported, it is accessible from anywhere on the machine using any of the model-based `classy` commands (e.g. `serve`, `predict`, etc).
:::
