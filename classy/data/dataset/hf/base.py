from typing import Optional, Callable, Iterator, List, Union

from transformers import AutoTokenizer

from classy.data.data_drivers import SequenceSample, TokensSample, SentencePairSample, QASample
from classy.data.dataset.base import BaseDataset
from classy.utils.log import get_project_logger
from classy.utils.vocabulary import Vocabulary

logger = get_project_logger(__name__)


class HFBaseDataset(BaseDataset):
    _shared_state = {}

    def __init__(
        self,
        samples_iterator: Callable[[], Iterator[Union[SequenceSample, SentencePairSample, TokensSample, QASample]]],
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
        batching_fields: Optional[List[str]] = None,
    ):
        if "tokenizer" not in self._shared_state:
            self._shared_state["tokenizer"] = AutoTokenizer.from_pretrained(
                transformer_model, use_fast=True, add_prefix_space=True
            )
        self.tokenizer = self._shared_state["tokenizer"]
        super().__init__(
            samples_iterator=samples_iterator,
            vocabulary=vocabulary,
            batching_fields=batching_fields or ["input_ids"],
            tokens_per_batch=tokens_per_batch,
            max_batch_size=max_batch_size,
            fields_batchers=None,
            section_size=section_size,
            prebatch=prebatch,
            materialize=materialize,
            min_length=min_length,
            max_length=max_length if max_length != -1 else self.tokenizer.model_max_length,
            for_inference=for_inference,
        )
        self._init_fields_batcher()
