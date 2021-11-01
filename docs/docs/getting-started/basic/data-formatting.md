---
sidebar_position: 2
title: Formatting your data
---

`classy` requires data to be formatted in a specific way according to the task you're tackling (check out [Tasks and Input Formats](/docs/reference-manual/tasks-and-formats) in the documentation).

In our case of **Named Entity Recognition** (i.e., *Token Classification*), we the data to be formatted in a way that each line represents a single sample.
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