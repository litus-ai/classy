---
sidebar_position: 0
title: classy Configuration
---

Behind the scenes, ` classy` works with `.yaml` configuration files to configure your experiment. This allows to modify
its behavior without having to change a single line of code! You can use `classy train [...] --print` to visualize the 
underlying configuration structure (there's actually more to it, but we'll come back later).

![Classy Train Print - Token](/img/intro/classy-train-print-tok.png)

If you want to change its values, you have 2 ways to go about it:
* Passing CLI overrides to classy train
* Using a profile

# Passing CLI overrides to classy train
You can change the config from the command line by using the `-c` parameter of `classy train`. Just pass 
`-c key1=value1 [...]` to `classy train` and that's it!

:::tip

For example, say you want to specify the min and max number of steps your train can run, you can do that by adding 
`-c trainer.pl_trainer.max_steps=X trainer.pl_trainer.min_steps=Y` to your `classy train` command.

:::

# Using a Profile
Applying changes from CLI allows for quick experimentation. However, sometimes you want to apply several changes to the 
underlying config, and doing it all from CLI is a mess. This is exactly why we devised the concept of a profile: it is 
a single file where you can specify all the changes you want to apply. We won't delve deeper into this for the moment, 
but we'll come back to this in a [couple of sections](/docs/getting-started/customizing-things/changing-profile).

:::tip

You can use profiles and CLI overrides together! Just remember that CLI overrides take precedence over everything else :)

:::