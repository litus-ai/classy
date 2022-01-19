from typing import Dict, List, Optional

from transformers import AutoTokenizer

from classy.data.dataset.base import BaseDataset
from classy.utils.log import get_project_logger

logger = get_project_logger(__name__)


class HFBaseDataset(BaseDataset):
    _shared_state = {}

    def __init__(
        self,
        transformer_model: str,
        additional_special_tokens: Optional[List[str]] = None,
        **kwargs,
    ):
        if "tokenizer" not in self._shared_state:
            self._shared_state["tokenizer"] = AutoTokenizer.from_pretrained(
                transformer_model,
                use_fast=True,
                add_prefix_space=True,
                additional_special_tokens=list(additional_special_tokens)
                if additional_special_tokens is not None
                else None,
            )
        self.tokenizer = self._shared_state["tokenizer"]
        batching_fields = (
            kwargs.pop("batching_fields")
            if "batching_fields" in kwargs
            else ["input_ids"]
        )
        super().__init__(
            fields_batchers=self.init_fields_batcher(),
            batching_fields=batching_fields,
            **kwargs,
        )

    def init_fields_batcher(self) -> Dict:
        raise NotImplementedError
