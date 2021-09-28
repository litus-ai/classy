from typing import Callable, List, Any, Dict, Union, Optional, Iterator, Generator

import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

from classy.data.data_drivers import (
    DataDriver,
    SentencePairSample,
    SequenceSample,
    TokensSample,
    QASample,
    GenerationSample,
)
from classy.utils.commons import chunks, flatten, add_noise_to_value
from classy.utils.log import get_project_logger
from classy.utils.vocabulary import Vocabulary

logger = get_project_logger(__name__)


def batchify(tensors: List[torch.Tensor], padding_value: int) -> torch.Tensor:
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value)


def batchify_matrices(tensors: List[torch.Tensor], padding_value: int) -> torch.Tensor:
    x = max([t.shape[0] for t in tensors])
    y = max([t.shape[1] for t in tensors])
    out_matrix = torch.zeros((len(tensors), x, y))
    out_matrix += padding_value
    for i, tensor in enumerate(tensors):
        out_matrix[i][0 : tensor.shape[0], 0 : tensor.shape[1]] = tensor
    return out_matrix


class BaseDataset(IterableDataset):
    @staticmethod
    def requires_vocab() -> bool:
        return True

    @staticmethod
    def fit_vocabulary(
        samples: Iterator[Union[SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample]]
    ) -> Vocabulary:
        raise NotImplementedError

    @classmethod
    def from_file(cls, path: str, data_driver: DataDriver, vocabulary: Vocabulary = None, **kwargs) -> "BaseDataset":

        if vocabulary is None and cls.requires_vocab():
            # vocabulary fitting here
            logger.info("Fitting vocabulary")
            vocabulary = cls.fit_vocabulary(data_driver.read_from_path(path))
            logger.info("Vocabulary fitting completed")

        return cls(samples_iterator=lambda: data_driver.read_from_path(path), vocabulary=vocabulary, **kwargs)

    @classmethod
    def from_lines(
        cls, lines: Iterator[str], data_driver: DataDriver, vocabulary: Vocabulary, **kwargs
    ) -> "BaseDataset":
        return cls(samples_iterator=lambda: data_driver.read(lines), vocabulary=vocabulary, **kwargs)

    @classmethod
    def from_samples(
        cls,
        samples: Iterator[Union[SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample]],
        vocabulary: Vocabulary,
        **kwargs,
    ):
        return cls(samples_iterator=lambda: samples, vocabulary=vocabulary, **kwargs)

    def __init__(
        self,
        samples_iterator: Callable[
            [], Iterator[Union[SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample]]
        ],
        vocabulary: Vocabulary,
        batching_fields: List[str],
        tokens_per_batch: int,
        max_batch_size: int,
        fields_batchers: Optional[Dict[str, Union[None, Callable[[list], Any]]]],
        section_size: int,
        prebatch: bool,
        materialize: bool,
        min_length: int,
        max_length: int,
        for_inference: bool,
    ):
        super().__init__()

        self.samples_iterator = samples_iterator
        self.vocabulary = vocabulary

        self.batching_fields = batching_fields
        assert len(batching_fields) > 0, "At least 1 batching field is required"

        self.tokens_per_batch, self.max_batch_size = tokens_per_batch, max_batch_size
        self.fields_batcher = fields_batchers
        self.prebatch, self.section_size = prebatch, section_size
        self.materialize = materialize
        self.min_length, self.max_length = min_length, max_length
        self.for_inference = for_inference

        if self.tokens_per_batch < self.max_length:
            logger.warning(
                f"Token batch size {self.tokens_per_batch} < max length {self.max_length}. This might result in batches with only 1 sample that contain more token than the specified token batch size"
            )

        # used to store the materialized dataset
        self._dataset_store = None
        if materialize:
            logger.warning("Materializing dataset.")
            self.materialize_dataset()

    def dataset_iterator_func(self):
        raise NotImplementedError

    def prebatch_elements(self, dataset_elements: List):
        dataset_elements = sorted(
            dataset_elements,
            key=lambda elem: add_noise_to_value(sum(len(elem[k]) for k in self.batching_fields), noise_param=0.1),
        )
        ds = list(chunks(dataset_elements, 512))
        np.random.shuffle(ds)
        return flatten(ds)

    def materialize_dataset(self) -> None:
        if self._dataset_store is not None:
            logger.info("The dataset is already materialized skipping materialization")
            return
        logger.info("Starting dataset materialization")
        self._dataset_store = list(self.dataset_iterator_func())
        logger.info("Materialization completed")

    def materialize_batches(self, dataset_elements: List[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:

        if self.prebatch:
            dataset_elements = self.prebatch_elements(dataset_elements)

        current_batch = []

        # function that creates a batch from the 'current_batch' list
        def output_batch() -> Dict[str, Any]:

            batch_dict = dict()

            de_values_by_field = {fn: [de[fn] for de in current_batch if fn in de] for fn in self.fields_batcher}

            # in case you provide fields batchers but in the batch there are no elements for that field
            de_values_by_field = {fn: fvs for fn, fvs in de_values_by_field.items() if len(fvs) > 0}

            assert len(set([len(v) for v in de_values_by_field.values()]))

            # todo: maybe we should report the user about possible fields filtering due to "None" instances
            de_values_by_field = {
                fn: fvs for fn, fvs in de_values_by_field.items() if all([fv is not None for fv in fvs])
            }

            for field_name, field_values in de_values_by_field.items():
                field_batch = (
                    self.fields_batcher[field_name](field_values)
                    if self.fields_batcher[field_name] is not None
                    else field_values
                )

                batch_dict[field_name] = field_batch

            return batch_dict

        max_len_discards, min_len_discards = 0, 0

        for de in dataset_elements:

            if self.max_batch_size != -1 and len(current_batch) == self.max_batch_size:
                yield output_batch()
                current_batch = []

            # todo: maybe here we want to check fields or stuff like that
            # some callback to filter out samples for example

            too_long_batching_fields = [
                k for k in self.batching_fields if self.max_length != -1 and len(de[k]) > self.max_length
            ]
            if len(too_long_batching_fields) > 0:
                max_len_discards += 1
                continue

            too_short_batching_fields = [
                k for k in self.batching_fields if self.min_length != -1 and len(de[k]) < self.min_length
            ]
            if len(too_short_batching_fields) > 0:
                min_len_discards += 1
                continue

            de_len = sum(len(de[k]) for k in self.batching_fields)

            future_max_len = max(
                de_len,
                max([sum(len(bde[k]) for k in self.batching_fields) for bde in current_batch], default=0),
            )

            future_tokens_per_batch = future_max_len * (len(current_batch) + 1)

            if (
                len(current_batch) > 0 and future_tokens_per_batch >= self.tokens_per_batch
            ):  # todo: add min batch size so as to support batching by size
                yield output_batch()
                current_batch = []

            current_batch.append(de)

        if len(current_batch) != 0:
            yield output_batch()

        if max_len_discards > 0:
            if self.for_inference:
                logger.warning(
                    f"WARNING: Inference mode is True but {max_len_discards} samples longer than max length were "
                    f"found. The {max_len_discards} samples will be DISCARDED. If you are doing some kind of evaluation"
                    f", this can INVALIDATE results. This might happen if the max length was not set to -1 or if the "
                    f"sample length exceeds the maximum length supported by the current model."
                )
            else:
                logger.warning(
                    f"During iteration, {max_len_discards} elements were "
                    f"discarded since longer than max length {self.max_length}"
                )

        if min_len_discards > 0:
            if self.for_inference:
                logger.warning(
                    f"WARNING: Inference mode is True but {min_len_discards} samples shorter than min length were "
                    f"found. The {min_len_discards} samples will be DISCARDED. If you are doing some kind of evaluation"
                    f", this can INVALIDATE results. This might happen if the min length was not set to -1 or if the "
                    f"sample length is shorter than the minimum length supported by the current model."
                )
            else:
                logger.warning(
                    f"During iteration, {min_len_discards} elements were "
                    f"discarded since shorter than min length {self.min_length}"
                )

    def __iter__(self):

        dataset_iterator = self.dataset_iterator_func() if self._dataset_store is None else self._dataset_store

        current_dataset_elements = []

        i = None
        for i, dataset_elem in enumerate(dataset_iterator, start=1):

            if len(current_dataset_elements) == self.section_size:
                for batch in self.materialize_batches(current_dataset_elements):
                    yield batch
                current_dataset_elements = []

            current_dataset_elements.append(dataset_elem)

            if i % 50_000 == 0:
                logger.info(f"Processed: {i} number of elements")

        if len(current_dataset_elements) != 0:
            for batch in self.materialize_batches(current_dataset_elements):
                yield batch

        if i is not None:
            logger.info(f"Dataset finished: {i} number of elements processed")
        else:
            logger.warning("Dataset empty")
