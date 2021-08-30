from typing import Optional, Callable, Iterable, Dict, Any, Union, Tuple, Iterator

import torch
from transformers import AutoTokenizer

from classy.data.dataset.base import BaseDataset, batchify
from classy.data.readers import SequenceReader, FileReader, SequenceSample

from classy.utils.vocabulary import Vocabulary


class HFSequenceDataset(BaseDataset):
    @classmethod
    def from_file(
        cls,
        path: str,
        sequence_reader: SequenceReader,
        **kwargs,
    ) -> "HFSequenceDataset":
        def r() -> Iterator[SequenceSample]:
            for sequence_sample in sequence_reader.read(path):
                yield sequence_sample

        return cls(samples_iterator=r, **kwargs)

    @staticmethod
    def fit_features_vocabulary(reader: FileReader, dataset_path: str) -> Optional[Vocabulary]:
        return None

    @staticmethod
    def fit_labels_vocabulary(reader: FileReader, dataset_path: str) -> Tuple[Optional[Vocabulary], Vocabulary]:
        return Vocabulary.from_samples(_l.label for _l in reader.read(dataset_path))

    def __init__(
        self,
        samples_iterator: Callable[[], Iterator[SequenceSample]],
        features_vocabulary: Optional[Vocabulary],
        labels_vocabulary: Optional[Vocabulary],
        transformer_model: str,
        tokens_per_batch: int,
        max_batch_size: Optional[int],
        main_field: str,
        section_size: int,
        prebatch: bool,
        shuffle: bool,
        max_length: int,
    ):
        super().__init__(
            features_vocabulary,
            labels_vocabulary,
            None,
            tokens_per_batch,
            max_batch_size,
            main_field,
            None,
            section_size,
            prebatch,
            shuffle,
            max_length,
        )
        self.samples_iterator = samples_iterator
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model, use_fast=True)
        self.__init_fields_batcher()

    def __init_fields_batcher(self) -> None:
        self.fields_batcher = {
            "input_ids": lambda lst: batchify(lst, padding_value=self.tokenizer.pad_token_id),
            "attention_mask": lambda lst: batchify(lst, padding_value=0),
            "labels": lambda lst: torch.tensor(lst, dtype=torch.long),
        }

    def dataset_iterator_func(self) -> Iterable[Dict[str, Any]]:
        for sequence_sample in self.samples_iterator():
            input_ids = self.tokenizer(sequence_sample.sequence)
            elem_dict = {
                "input_ids": self.tokenizer(sequence_sample.sequence),
                "attention_mask": torch.ones_like(input_ids),
            }
            if sequence_sample.label is not None:
                elem_dict["label"] = self.labels_vocabulary.get_idx(sequence_sample.label)
            yield elem_dict
