---
sidebar_position: 3
title: Choosing a profile
---

:::tip
This step is **not mandatory**, but we highly recommend you to read it as it touches an important component of
`classy`, the profiles, which is needed in case you want to heavily modify your training configuration.

:::

It might be the case that you have constraints of any sort (hardware, performance-wise, etc.), and you might
be interested in knowing how to change the default underlying model / optimizer used to train in order to either
fit in smaller GPUs, be faster, or achieve higher accuracy.

In `classy`, this is achieved through *Profiles*, which a user can employ as a way of changing the training configuration
of their model to fit different criteria.

`classy` comes with a predefined set of profiles, which you can find [here](/docs/reference-manual/profiles/).
The list includes the underlying transformer model, optimizer and a few key features that each profile shines for.

For this tutorial, we'll stick with a fast yet powerful model, *DistilBERT*.
