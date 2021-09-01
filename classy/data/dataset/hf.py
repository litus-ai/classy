import random
from typing import Optional, Callable, Iterable, Dict, Any, Union, Tuple, Iterator

import logging
import torch
from transformers import AutoTokenizer

from classy.data.dataset.base import BaseDataset, batchify
from classy.data.data_drivers import SequenceDataDriver, DataDriver, SequenceSample
from classy.utils.log import get_project_logger

from classy.utils.vocabulary import Vocabulary


logger = get_project_logger(__name__)


class HFSequenceDataset(BaseDataset):
    @classmethod
    def from_file(
        cls, path: str, data_driver: DataDriver, vocabulary: Optional[Dict[str, Vocabulary]] = None, **kwargs
    ) -> "BaseDataset":
        def r() -> Iterator[SequenceSample]:
            for sequence_sample in data_driver.read_from_path(path):
                yield sequence_sample

        if vocabulary is None:
            # vocabulary fitting here
            logger.info("Fitting vocabulary")
            vocabulary = Vocabulary.from_samples([{"labels": _l.label} for _l in data_driver.read_from_path(path)])
            logger.info("Vocabulary fitting completed")

        return cls(samples_iterator=r, vocabulary=vocabulary, **kwargs)

    @classmethod
    def from_lines(
        cls, lines: Iterable[str], data_driver: DataDriver, vocabulary: Optional[Dict[str, Vocabulary]] = None, **kwargs
    ) -> "BaseDataset":
        def r() -> Iterator[SequenceSample]:
            for sequence_sample in data_driver.read(lines):
                yield sequence_sample

        if vocabulary is None:
            # vocabulary fitting here
            logger.info("Fitting vocabulary")
            vocabulary = Vocabulary.from_samples([{"labels": _l.label} for _l in data_driver.read_from_path(path)])
            logger.info("Vocabulary fitting completed")

        return cls(samples_iterator=r, vocabulary=vocabulary, **kwargs)

    def __init__(
        self,
        samples_iterator: Callable[[], Iterator[SequenceSample]],
        vocabulary: Vocabulary,
        transformer_model: str,
        tokens_per_batch: int,
        max_batch_size: Optional[int],
        section_size: int,
        prebatch: bool,
        shuffle: bool,
        min_length: int,
        max_length: int,
    ):
        super().__init__(
            dataset_iterator_func=None,
            batching_fields=["input_ids"],
            tokens_per_batch=tokens_per_batch,
            max_batch_size=max_batch_size,
            fields_batchers=None,
            section_size=section_size,
            prebatch=prebatch,
            shuffle=shuffle,
            min_length=min_length,
            max_length=max_length,
        )
        self.vocabulary = vocabulary
        self.samples_iterator = samples_iterator
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model, use_fast=True)
        self.__init_fields_batcher()

    def __init_fields_batcher(self) -> None:
        self.fields_batcher = {
            "input_ids": lambda lst: batchify(lst, padding_value=self.tokenizer.pad_token_id),
            "attention_mask": lambda lst: batchify(lst, padding_value=0),
            "labels": lambda lst: torch.tensor(lst, dtype=torch.long),
            "samples": None,
        }

    def dataset_iterator_func(self) -> Iterable[Dict[str, Any]]:

        for sequence_sample in self.samples_iterator():
            input_ids = self.tokenizer(sequence_sample.sequence, return_tensors="pt")["input_ids"][0]
            elem_dict = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
            if sequence_sample.label is not None:
                elem_dict["labels"] = [self.vocabulary.get_idx(k="labels", elem=sequence_sample.label)]
            elem_dict["samples"] = sequence_sample
            yield elem_dict
