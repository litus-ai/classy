---
sidebar_position: 5
title: Custom Dataset
---


Implementing your custom dataset with classy is easy. You just need to subclass BaseDataset:

```python
class MyCustomDataset(BaseDataset):
    @staticmethod
    def requires_vocab() -> bool:
        # returns true if the dataset requires fitting a vocabulary, false otherwise
        pass

    @staticmethod
    def fit_vocabulary(
        samples: Iterator[
            Union[
                SentencePairSample,
                SequenceSample,
                TokensSample,
                QASample,
                GenerationSample,
            ]
        ]
    ) -> Vocabulary:
        # fits the vocabulary
        pass

    def __init__(self, *args, **kwargs):
        # construct fields batchers
        fields_batchers = {...}
        super().__init__(*args, fields_batchers=fields_batchers, **kwargs)

    def dataset_iterator_func(self) -> Iterable[Dict[str, Any]]:
        # yields a sequence of dictionaries, each representing a sample
        pass
```

The underlying flow is as follows:
* Dataset instantiation is transparent to you, and takes place via 1 of 3 class methods:
    * BaseDataset.from_file
    * BaseDataset.from_lines
    * BaseDataset.from_samples
* Regardless on which one is invoked, BaseDataset exposes to you a *samples_iterator* function that, once invoked, returns a sequence of classy samples
* In your *dataset_iterator_func*, you iterate on these samples and convert them to dictionary objects
* These dictionary-like samples are then batched using the *fields_batchers* variable you pass to BaseDataset in your *\_\_init\_\_*; it is essentially a dictionary mapping
  keys in your dictionary-like samples to collating functions

## A Minimal Example

Practically, imagine you want to build your own SequenceDataset for BERT.

```python title="classy.data.dataset.my_bert_sequence_dataset.py"
from transformers import AutoTokenizer
from classy.data.data_drivers import SequenceSample
from classy.data.dataset.base import batchify, BaseDataset


class MyBertSequenceDataset(BaseDataset):
    pass
```

You first deal with the vocabulary methods. As you are doing sequence classification, you need to fit the label vocabulary:

```python
@staticmethod
def requires_vocab() -> bool:
    return True


@staticmethod
def fit_vocabulary(samples: Iterator[SequenceSample]) -> Vocabulary:
    return Vocabulary.from_samples(
        [{"labels": sample.reference_annotation} for sample in samples]
    )
```

Then, define your constructor and, in particular, your *fields_batchers*:

```python
def __init__(
    self,
    samples_iterator: Callable[
        [], Iterator[Union[SequenceSample, SentencePairSample, TokensSample, QASample]]
    ],
    vocabulary: Vocabulary,
    transformer_model: str,
    tokens_per_batch: int,
    max_batch_size: Optional[int],
    section_size: int,
    prebatch: bool,
    materialize: bool,
    min_length: int,
    max_length: int,
    for_inference: bool,
):

    # load bert tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(
        transformer_model, use_fast=True, add_prefix_space=True
    )

    # define fields_batchers
    fields_batcher = {
        "input_ids": lambda lst: batchify(
            lst, padding_value=self.tokenizer.pad_token_id
        ),
        "attention_mask": lambda lst: batchify(lst, padding_value=0),
        "labels": lambda lst: torch.tensor(lst, dtype=torch.long),
        "samples": None,
    }

    super().__init__(
        samples_iterator=samples_iterator,
        vocabulary=vocabulary,
        batching_fields=["input_ids"],
        tokens_per_batch=tokens_per_batch,
        max_batch_size=max_batch_size,
        fields_batchers=fields_batcher,
        section_size=section_size,
        prebatch=prebatch,
        materialize=materialize,
        min_length=min_length,
        max_length=max_length if max_length != -1 else self.tokenizer.model_max_length,
        for_inference=for_inference,
    )
```

Finally, you need to implement the *dataset_iterator_func*:

```python
def dataset_iterator_func(self) -> Iterable[Dict[str, Any]]:
    # iterate on samples
    for sequence_sample in self.samples_iterator():
        # invoke tokenizer
        input_ids = self.tokenizer(sequence_sample.sequence, return_tensors="pt")[
            "input_ids"
        ][0]
        # build dict
        elem_dict = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
        if sequence_sample.reference_annotation is not None:
            # use vocabulary to convert string labels to int labels
            elem_dict["labels"] = [
                self.vocabulary.get_idx(
                    k="labels", elem=sequence_sample.reference_annotation
                )
            ]
        elem_dict["samples"] = sequence_sample
        yield elem_dict
```
