---
sidebar_position: 4
title: Mixins
---

To extend the functionalities of our models, we use the **mixin** pattern. Basically, a mixin is a self-contained unit
that offers a number of predefined methods that are added to the object that is *mixed-in*.

In `classy`, we have four mixins:

- `PredictionMixin`
- `SavingMixin`
- `TaskMixin`
- `TaskUIMixin`

`SavingMixin` offers a utility to save and load a few samples from your dataset to be used inside `classy demo`.

`PredictionMixin` offers a method that is used at inference time, `batch_predict`, that your custom `ClassyPLModule` **should always implement**.

As for `TaskMixin` and `TaskUIMixin`, they handle the behaviour of a model when invoked through either `classy predict interactive`
or `classy demo`. `TaskMixin` handles how your model reads input from the command line (e.g., a sentence-pair classifier
will have to read from input twice, as opposed to, for example, a sequence classifier); similarly, `TaskUIMixin` handles
your model's rendering of its inputs and outputs in `classy demo`, hence through [Streamlit](https://streamlit.io/).

:::tip

By default, `ClassyPLModule` is defined as a `pl.LightningModule` mixed in with `PredictionMixin` and `SavingMixin`. Our
default task-specific models (`HF<Task>PLModule`) are mixed in with their task-specific bash and UI mixins
(e.g., `HFTokensPLModule` is `ClassyModule` mixed in with `TokensTaskMixin` and `TokensTaskUIMixin`).

:::