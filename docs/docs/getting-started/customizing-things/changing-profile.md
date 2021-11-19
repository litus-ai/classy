---
sidebar_position: 2
title: Changing Profile
---

Profiles are `classy`'s way of handling custom configurations. There are two ways in which you can use a profile:
1. using a single profile file (e.g., `configurations/profiles/custom.yaml`), which can override every single pre-existing configuration value;
2. using a profile file in combination with `hydra`'s config groups.


:::caution
Your custom profiles **must be** stored under `configurations/profiles`, or `classy` won't be able to discover them.
:::


## Single-file profile
Say that you are implementing a custom Named Entity Recognition model. In your profile, you are going to specify
your model's configuration explicitly, and it will be overridden over `classy`'s default.

```yaml title=configurations/profiles/single-file.yaml
model:
  _target_: src.model.MyCustomClassyNERModel
  parameter1: ...
  parameter2: ...
```

Then, to train using your model, you just run `classy train token data/ner -n custom-experiment --profile single-file`.

:::info

This kind of composition is very similar to `hydra`'s default behaviour, but with an important difference: when using
profiles, configurations are not additive but exclusive (if they have a `_target_` property): for example, 
if your `model` defines an `optim_conf`, such configuration will **entirely replace** the previously existing 
`optim_conf`, instead of adding its config parameters to it.

:::

## Exploiting config groups

Single file configs are prone to become very large very quickly. Thus, we extend `hydra`'s behaviour to support config
groups specification even outside the `defaults` list. In a nutshell, this means that you can have a config file under 
`configurations/models/custom-model.yaml` that is referred by `configurations/profiles/grouped.yaml`.

```yaml title=configurations/models/custom-model.yaml
model:
  _target_: src.model.MyCustomClassyNERModel
  parameter1: ...
  parameter2: ...
```


```yaml title=configurations/profiles/grouped.yaml
model: custom-model
```
