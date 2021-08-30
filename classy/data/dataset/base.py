import logging
from typing import Callable, Iterable, List, Any, Dict, Union, Optional, Tuple, Iterator

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

from classy.data.readers import FileReader
from classy.utils.commons import chunks, flatten
from classy.utils.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


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
    @classmethod
    def from_file(
        cls,
        path: str,
        file_reader: FileReader,
        **kwargs,
    ) -> "BaseDataset":
        raise NotImplementedError

    @staticmethod
    def fit_features_vocabulary(reader: FileReader, dataset_path: str) -> Optional[Vocabulary]:
        raise NotImplementedError

    @staticmethod
    def fit_labels_vocabulary(reader: FileReader, dataset_path: str) -> Tuple[Optional[Vocabulary], Vocabulary]:
        raise NotImplementedError

    def __init__(
        self,
        features_vocabulary: Vocabulary,
        labels_vocabulary: Vocabulary,
        dataset_iterator_func: Optional[Callable[[], Iterable[Dict[str, Any]]]],
        tokens_per_batch: int,
        max_batch_size: Optional[int],
        main_field: str,
        fields_batchers: Optional[Dict[str, Union[None, Callable[[list], Any]]]],
        section_size: int,
        prebatch: bool,
        shuffle: bool,
        max_length: int,
    ):
        super().__init__()

        # vocabularies
        self.features_vocabulary = (
            features_vocabulary
            if features_vocabulary is not None
            else self.fit_features_vocabulary(self.sequence_reader, self.path)
        )
        self.labels_vocabulary = (
            labels_vocabulary
            if labels_vocabulary is not None
            else self.fit_labels_vocabulary(self.sequence_reader, self.path)
        )

        # you can subclass TokenBasedDataset in this way
        if dataset_iterator_func is not None:
            self.dataset_iterator_func = dataset_iterator_func

        self.tokens_per_batch = tokens_per_batch
        self.max_batch_size = max_batch_size
        self.main_field = main_field
        self.fields_batcher = fields_batchers
        self.section_size = section_size
        self.prebatch = prebatch
        self.shuffle = shuffle
        self.max_length = max_length

        if self.shuffle and not self.prebatch:
            logger.warning("If you set prebatch to False the shuffle parameters has no effect")

    def prebatch_elements(self, dataset_elements: list) -> list:
        # todo: too many magic numbers in this block of code.
        if self.shuffle:
            dataset_elements = sorted(
                dataset_elements,
                key=lambda de: len(de[self.main_field]) + torch.randint(0, 10, (1,)),
            )
            dataset_elements = list(chunks(dataset_elements, 2048))
            np.random.shuffle(dataset_elements)
            dataset_elements = flatten(dataset_elements)
        else:
            dataset_elements = sorted(dataset_elements, key=lambda de: len(de[self.main_field]))

        return dataset_elements

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

        for de in dataset_elements:

            if self.max_batch_size is not None and len(current_batch) == self.max_batch_size:
                batches.append(output_batch())
                current_batch = []

            de_main_len = len(de[self.main_field])

            # todo: maybe here we want to check fields or stuff like that
            # some callback to filter out samples for example

            if de_main_len > self.max_length:
                logger.warning(f"Discarding element: max length exceeded ({de_main_len} > {self.max_length})")
                continue

            if de_main_len > self.tokens_per_batch:
                logger.warning(
                    f'Discarding element: length greater than "tokens per batch"'
                    f" ({de_main_len} > {self.tokens_per_batch})"
                )
                continue

            future_max_len = max(
                de_main_len,
                max([len(bde[self.main_field]) for bde in current_batch], default=0),
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

        current_dataset_elements = []

        for i, dataset_elem in enumerate(self.dataset_iterator_func()):

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

        logger.info(f"Dataset finished: {i} number of elements processed")
