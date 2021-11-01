---
sidebar_position: 3
title: Choosing a profile
---

:::tip
This step is **not mandatory**, but we highly recommend you to read it as it touches an important component of 
`classy`, the profiles, which is needed in case you want to heavily modify training configuration. 

:::

It might be the case that you have constraints of any sort (hardware, performance-wise, etc.), and you might
be interested in knowing how to change the default underlying model / optimizer used to train in order to either
fit in smaller GPUs, be faster, or achieve higher accuracy. 

In `classy`, this is achieved through *Profiles*, which a user can employ as a way of changing the training configuration
of their model to fit different criteria.

`classy` comes with a predefined set of profiles, which you can find [here](/docs/reference-manual/profiles/).
The list includes the underlying transformer model, optimizer and a few key features that each profile shines for.

## Visualizing your Configuration

`classy train` has an option that lets you visualize the full materialized configuration in your terminal, showing you 
where each component comes from (including overridden configuration groups, overrides, profiles, etc.). 
To print it, simply append `--print` to your current `classy train` command!

![Classy Train Print - Token](/img/intro/classy-train-print-tok.png)
