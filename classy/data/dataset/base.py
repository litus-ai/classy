import random
from typing import Callable, List, Any, Dict, Union, Optional, Iterator

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from tqdm import tqdm

from classy.data.data_drivers import DataDriver, SentencePairSample, SequenceSample, TokensSample
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
    def fit_vocabulary(samples: Iterator[Union[SentencePairSample, SequenceSample, TokensSample]]) -> Vocabulary:
        raise NotImplementedError

    @classmethod
    def from_file(
            cls, path: str, data_driver: DataDriver, vocabulary: Vocabulary = None, **kwargs
    ) -> "BaseDataset":
        def r() -> Iterator[Union[SentencePairSample, SequenceSample, TokensSample]]:
            for sequence_sample in data_driver.read_from_path(path):
                yield sequence_sample

        if vocabulary is None:
            # vocabulary fitting here
            logger.info("Fitting vocabulary")
            vocabulary = cls.fit_vocabulary(data_driver.read_from_path(path))
            logger.info("Vocabulary fitting completed")

        return cls(
            samples_iterator=r,
            vocabulary=vocabulary,
            **kwargs
        )

    @classmethod
    def from_lines(
            cls, lines: Iterator[str], data_driver: DataDriver, vocabulary: Vocabulary, **kwargs
    ) -> "BaseDataset":
        def r() -> Iterator[Union[SentencePairSample, SequenceSample, TokensSample]]:
            for sequence_sample in data_driver.read(lines):
                yield sequence_sample

        return cls(samples_iterator=r, vocabulary=vocabulary, **kwargs)

    def __init__(
        self,
        samples_iterator: Callable[[], Iterator[Union[SentencePairSample, SequenceSample, TokensSample]]],
        vocabulary: Vocabulary,
        batching_fields: List[str],
        tokens_per_batch: int,
        max_batch_size: int,
        fields_batchers: Optional[Dict[str, Union[None, Callable[[list], Any]]]],
        section_size: int,
        prebatch: bool,
        shuffle: bool,
        min_length: int,
        max_length: int,
    ):
        super().__init__()

        self.samples_iterator = samples_iterator
        self.vocabulary = vocabulary

        self.batching_fields = batching_fields
        assert len(batching_fields) > 0, "At least 1 batching field is required"

        self.tokens_per_batch, self.max_batch_size = tokens_per_batch, max_batch_size
        self.fields_batcher = fields_batchers
        self.prebatch, self.section_size = prebatch, section_size
        self.shuffle = shuffle
        self.min_length, self.max_length = min_length, max_length

    def dataset_iterator_func(self):
        raise NotImplementedError

    def prebatch_elements(self, dataset_elements: List):
        dataset_elements = sorted(
            dataset_elements,
            key=lambda elem: add_noise_to_value(sum(len(elem[k]) for k in self.batching_fields), noise_param=0.1),
        )
        ds = list(chunks(dataset_elements, 512))
        random.shuffle(ds)
        return flatten(ds)

    def materialize_batches(self, dataset_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        if self.prebatch:
            dataset_elements = self.prebatch_elements(dataset_elements)

        batches = []
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
                batches.append(output_batch())
                current_batch = []

            # todo: maybe here we want to check fields or stuff like that
            # some callback to filter out samples for example

            too_long_batching_fields = [
                k for k in self.batching_fields if self.max_length != -1 and len(de[k]) > self.max_length
            ]
            if len(too_long_batching_fields) > 0:
                max_len_discards += 1
                if max_len_discards % 10 == 0:
                    # if max_len_discards % 1_000 == 0:
                    logger.warning(
                        f"{max_len_discards} elements discarded since longer than max length {self.max_length}"
                    )
                continue

            too_short_batching_fields = [
                k for k in self.batching_fields if self.min_length != -1 and len(de[k]) < self.min_length
            ]
            if len(too_short_batching_fields) > 0:
                min_len_discards += 1
                if min_len_discards % 10 == 0:
                    # if min_len_discards % 1_000 == 0:
                    logger.warning(
                        f"{max_len_discards} elements discarded since shorter than max length {self.min_length}"
                    )
                continue

            de_len = sum(len(de[k]) for k in self.batching_fields)

            if de_len > self.tokens_per_batch:
                logger.warning(
                    f'Discarding element: length greater than "tokens per batch"'
                    f" ({de_len} > {self.tokens_per_batch})"
                )
                continue

            future_max_len = max(
                de_len,
                max([sum(len(bde[k]) for k in self.batching_fields) for bde in current_batch], default=0),
            )

            future_tokens_per_batch = future_max_len * (len(current_batch) + 1)

            if (
                future_tokens_per_batch >= self.tokens_per_batch
            ):  # todo: add min batch size so as to support batching by size
                batches.append(output_batch())
                current_batch = []

            current_batch.append(de)

        if len(current_batch) != 0:
            batches.append(output_batch())

        return batches

    def __iter__(self):

        dataset_iterator = self.dataset_iterator_func()
        if self.shuffle:
            logger.warning(
                "Careful: shuffle is set to true and requires materializing the ENTIRE dataset into memory"
            )
            dataset_iterator = list(tqdm(dataset_iterator, desc="Materializing dataset"))
            logger.info("Materliziation completed, now shuffling")
            random.shuffle(dataset_iterator)
            logger.info("Shuffling completed")

        current_dataset_elements = []

        i = None
        for i, dataset_elem in enumerate(dataset_iterator):

            if len(current_dataset_elements) == self.section_size:
                for batch in self.materialize_batches(current_dataset_elements):
                    yield batch
                current_dataset_elements = []

            current_dataset_elements.append(dataset_elem)

            if i % 100_000 == 0:
                logger.info(f"Processed: {i} number of elements")

        if len(current_dataset_elements) != 0:
            for batch in self.materialize_batches(current_dataset_elements):
                yield batch

        if i is not None:
            logger.info(f"Dataset finished: {i} number of elements processed")
        else:
            logger.warning("Dataset empty")
