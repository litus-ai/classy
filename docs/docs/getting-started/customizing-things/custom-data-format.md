---
sidebar_position: 3
title: Custom Data Format
---

By default, classy only supports .tsv and .jsonl files. However, you can easily add support for your own file format on some task.
You just need to implement your own data driver and register it:

```python
# implement your data driver
class CustomDataDriver(DataDriver):
    def read(self, lines: Iterator[str]) -> Generator[Union[SequenceSample, SentencePairSample, TokensSample, QASample, GenerationSample], None, None]:
        raise NotImplementedError

    def save(self, samples: Iterator[Union[SequenceSample, SentencePairSample, TokensSample, QASample, GenerationSample]], path: str):
        raise NotImplementedError

# register it
READERS_DICT[(YOUR_TASK, YOUR_FILE_EXTENSION)] = CustomDataDriver
```

:::caution

`classy` uses the tuple (task, file-extension) to determine the data driver to instantiate for some file. This means that
postpending file extensions is mandatory, even on Unix systems.

:::

## A Minimal Example

For instance, imagine you were to reimplement the `.jsonl` data driver for Sequence Classification:

```python
class JSONLSequenceDataDriver(SequenceDataDriver):
    pass
```

:::info

SequenceDataDriver is just a subclass of DataDriver where the sample types have been downcasted to SequenceSample only.

:::

You would first implement the read method:

```python
def read(self, lines: Iterator[str]) -> Generator[SequenceSample, None, None]:
    # iterate on lines
    for line in lines:
        # read json object and instantiate sequence sample
        yield SequenceSample(**json.loads(line))
```

and, then, the save method:

```python
def save(self, samples: Iterator[SequenceSample], path: str):
    with open(path, "w") as f:
        # iterate on samples
        for sample in samples:
            # dump json object
            f.write(json.dumps({"sequence": sample.sequence, "label": sample.label}) + "\n")
```

:::tip

While both `.jsonl` and `.tsv` are one-sample-per-line formats, your own data driver does not need to follow this behavior. As you
have access to the lines iterator, you can read your file as you see fit.

:::