---
sidebar_position: 5
title: describe
---

:::info

You need to have installed `classy` with describe support. You can do this with `pip install classy-core[describe]`

:::

`classy describe` is a data analysis tool with general and task-specific stats over your dataset.

Say that you have a Sequence Classification dataset, and you want some preliminary information about the data itself
(for example the distribution of the number of characters in the sentences), `classy describe` is the tool for you!

:::info

Depending on the task, `classy describe` produces different reports! For example, in Token Classification you can see the
distribution of tokens in your dataset.

:::

:::tip

`classy describe` has an optional parameter `--tokenize <lang>` that tokenizes your dataset using the `<lang>`-specific
**moses tokenizer**. You can use it with every task except Token Classification!

:::

```bash
classy describe --dataset data/sequence/sst2/train.jsonl sequence
```


![Classy Describe Sequence - Characters Distribution](/img/intro/classy-describe-seq-chars.png)

![Classy Describe Sequence - Labels Distribution](/img/intro/classy-describe-seq-labels.png)
