---
sidebar_position: 2
title: Organizing your data
---

## Data Formatting

`classy` requires data to be formatted in a specific way according to the task you're tackling (check out [Tasks and Input Formats](/docs/reference-manual/tasks-and-formats) in the documentation).

In our case of **Named Entity Recognition** (i.e., *Token Classification*), we need the data to be formatted in a way that each line represents a single sample.
For instance, taking again our running example of *Barack Obama visited Google in California*, we can format it as follows:

```text
Barack Obama visited Google in California\tPER PER O ORG O LOC
```

That is, a TSV (tab-separated values) file which has a space-separated sequence of tokens as the first column, and
a space-separated sequence of labels as the second column (both sequences **must have** the same number of elements).

:::tip
`classy` by default supports `.tsv` and `.jsonl` as input formats (see [the documentation](/docs/reference-manual/tasks-and-formats)),
but you can [add custom formats](/docs/getting-started/customizing-things/custom-data-format/) as well.
:::

If your dataset is already formatted like this, great! Otherwise, this is the only bit where coding is required.
You can either convert it yourself (via a python or bash script, whatever you're comfortable with), or you can register
a [custom data reader](/docs/getting-started/customizing-things/custom-data-format/) to support your dataset format.


## Organizing Datsets
In `classy`, as in standard machine learning projects, the most simple way to organize your datasets is to create
a directory containing the train, validation and test datasets.
```
data/ner-data
├── train.tsv
├── validation.tsv
└── test.tsv
```

In this way, `classy` will automatically infer the splits of your dataset from the directory structure.

:::tip
If you have multiple training files, or you want to specify the splits using a different directory structure, you can
use a _training coordinates_ file. You can find a complete guide on how to do it in the
[Reference Manual](/docs/reference-manual/cli/train/).
:::
