import collections
import re
from typing import Any, Callable, Dict, Generator, Iterator, List

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import MBartTokenizerFast
from transformers.models.mbart.tokenization_mbart_fast import FAIRSEQ_LANGUAGE_CODES

from classy.data.data_drivers import ClassySample
from classy.data.dataset.base import batchify
from classy.data.dataset.hf.base import HFBaseDataset
from classy.utils.log import get_project_logger
from classy.utils.vocabulary import Vocabulary

logger = get_project_logger(__name__)


def resolve_hf_generation_base_dataset_on_transformer_model(
    transformer_model: str,
) -> str:
    if re.fullmatch("facebook/bart-(base|large)", transformer_model):
        return "classy.data.dataset.hf.generation.BartHFGenerationDataset.from_file"
    elif re.fullmatch("facebook/mbart-large-(cc25|50)", transformer_model):
        return "classy.data.dataset.hf.generation.MBartHFGenerationDataset.from_file"
    elif transformer_model.startswith("gpt2"):
        return "classy.data.dataset.hf.generation.GPT2HFGenerationCataset.from_file"
    else:
        raise ValueError(
            f"{transformer_model} not currently supported in automatic resolution. But you can still write your own dataset (write _target_ and its parameters)."
        )


OmegaConf.register_new_resolver(
    "resolve_hf_generation_base_dataset_on_transformer_model",
    resolve_hf_generation_base_dataset_on_transformer_model,
)


class HFGenerationBaseDataset(HFBaseDataset):
    @staticmethod
    def requires_vocab() -> bool:
        return False

    @staticmethod
    def fit_vocabulary(samples: Iterator[ClassySample]) -> Vocabulary:
        raise NotImplementedError

    def __init__(self, **kwargs):
        super().__init__(
            batching_fields=self.get_batching_fields(kwargs["for_inference"]), **kwargs
        )

    def get_batching_fields(self, inference_mode: bool):
        raise NotImplementedError

    def materialize_batches(
        self, dataset_elements: List[Dict[str, Any]]
    ) -> Generator[Dict[str, Any], None, None]:
        for group in self.group_elements_on_materializations(
            dataset_elements, inference_mode=self.for_inference
        ):
            yield from super().materialize_batches(group)

    def group_elements_on_materializations(
        self, dataset_elements: List[Dict[str, Any]], inference_mode: bool
    ) -> List[List[Dict[str, Any]]]:
        return [dataset_elements]


class EncDecHFGenerationBaseDataset(HFGenerationBaseDataset):
    def get_batching_fields(self, inference_mode: bool) -> List[str]:
        return ["input_ids", "labels"] if not inference_mode else ["input_ids"]


class DecHFGenerationBaseDataset(HFGenerationBaseDataset):
    def get_batching_fields(self, inference_mode: bool) -> List[str]:
        return ["input_ids"]

    def group_elements_on_materializations(
        self, dataset_elements: List[Dict[str, Any]], inference_mode: bool
    ) -> List[List[Dict[str, Any]]]:

        if not inference_mode:
            return [dataset_elements]

        groups = collections.defaultdict(list)

        for de in dataset_elements:
            de_len = len(de["input_ids"])
            groups[de_len].append(de)

        return [group for group_len, group in groups.items()]


class BartHFGenerationDataset(EncDecHFGenerationBaseDataset):
    def init_fields_batcher(self) -> Dict:
        return {
            "input_ids": lambda lst: batchify(
                lst, padding_value=self.tokenizer.pad_token_id
            ),
            "attention_mask": lambda lst: batchify(lst, padding_value=0),
            "samples": None,
            "labels": lambda lst: batchify(
                lst, padding_value=-100
            ),  # -100 == cross entropy ignore index
            "decoder_attention_mask": lambda lst: batchify(lst, padding_value=0),
            "decoder_start_token_id": lambda lst: batchify(
                lst, padding_value=-1
            ),  # -1 is to force the model to crash (padding should never happen)
        }

    def dataset_iterator_func(self):
        for sample in self.samples_iterator():

            assert (
                sample.source_language is None and sample.target_language is None
            ), f"BartHFGenerationSampleEncoder does not support language specification"

            tokenization_output = self.tokenizer(
                sample.source_sequence,
                return_tensors="pt",
                **(
                    {"truncation": True, "max_length": self.max_length}
                    if self.truncation
                    else {}
                ),
            )
            elem_dict = {
                "input_ids": tokenization_output["input_ids"].squeeze(),
                "attention_mask": tokenization_output["attention_mask"].squeeze(),
            }

            if not self.for_inference:
                if sample.reference_annotation is not None:
                    tokenization_output = self.tokenizer(
                        sample.reference_annotation,
                        return_tensors="pt",
                        **(
                            {"truncation": True, "max_length": self.max_length}
                            if self.truncation
                            else {}
                        ),
                    )
                    elem_dict.update(
                        **{
                            "labels": tokenization_output["input_ids"].squeeze(),
                            "decoder_attention_mask": tokenization_output[
                                "attention_mask"
                            ].squeeze(),
                        }
                    )
            else:
                elem_dict["decoder_start_token_id"] = torch.tensor(
                    [self.tokenizer.eos_token_id]
                )

            elem_dict["samples"] = sample
            yield elem_dict


class MBartHFGenerationDataset(BartHFGenerationDataset):

    mbart_l2l_code = {}
    for l_code in FAIRSEQ_LANGUAGE_CODES:
        l = l_code[: l_code.index("_")]
        mbart_l2l_code[l] = l_code

    def dataset_iterator_func(self):
        for sample in self.samples_iterator():
            assert (
                sample.source_language is not None
                and sample.target_language is not None
            ), f"MBARTHFGenerationSampleEncoder requires language specification"

            self.tokenizer: MBartTokenizerFast
            self.tokenizer.src_lang = self.mbart_l2l_code.get(
                sample.source_language, sample.source_language
            )
            self.tokenizer.tgt_lang = self.mbart_l2l_code.get(
                sample.target_language, sample.target_language
            )

            tokenization_output = self.tokenizer(
                sample.source_sequence,
                return_tensors="pt",
                **(
                    {"truncation": True, "max_length": self.max_length}
                    if self.truncation
                    else {}
                ),
            )
            elem_dict = {
                "input_ids": tokenization_output["input_ids"].squeeze(),
                "attention_mask": tokenization_output["attention_mask"].squeeze(),
            }

            if not self.for_inference:
                with self.tokenizer.as_target_tokenizer():
                    tokenization_output = self.tokenizer(
                        sample.reference_annotation,
                        return_tensors="pt",
                        **(
                            {"truncation": True, "max_length": self.max_length}
                            if self.truncation
                            else {}
                        ),
                    )
                    elem_dict.update(
                        **{
                            "labels": tokenization_output["input_ids"].squeeze(),
                            "decoder_attention_mask": tokenization_output[
                                "attention_mask"
                            ].squeeze(),
                        }
                    )
            else:
                elem_dict["decoder_start_token_id"] = torch.tensor(
                    [
                        self.tokenizer.convert_tokens_to_ids(
                            self.mbart_l2l_code.get(
                                sample.target_language, sample.target_language
                            )
                        )
                    ]
                )

            elem_dict["samples"] = sample
            yield elem_dict

    def group_elements_on_materializations(
        self, dataset_elements: List[Dict[str, Any]], inference_mode: bool
    ) -> List[List[Dict[str, Any]]]:
        if not inference_mode:
            return [dataset_elements]

        groups = collections.defaultdict(list)

        for de in dataset_elements:
            groups[de["samples"].target_language].append(de)

        return [group for group_len, group in groups.items()]


class GPT2HFGenerationCataset(DecHFGenerationBaseDataset):
    def init_fields_batcher(self) -> Dict[str, Callable]:
        return {
            "input_ids": lambda lst: batchify(
                lst, padding_value=0
            ),  # todo gpt2 does not seem to have a pad token id, double check solution
            "attention_mask": lambda lst: batchify(lst, padding_value=0),
            "samples": None,
            "labels": lambda lst: batchify(
                lst, padding_value=-100
            ),  # -100 == cross entropy ignore index
        }

    def dataset_iterator_func(self):
        for sample in self.samples_iterator():

            tokenization_output = self.tokenizer(
                sample.source_sequence,
                return_tensors="pt",
                **(
                    {"truncation": True, "max_length": self.max_length}
                    if self.truncation
                    else {}
                ),
            )
            elem_dict = {
                "input_ids": tokenization_output["input_ids"].squeeze(),
                "attention_mask": tokenization_output["attention_mask"].squeeze(),
                "samples": sample,
            }
            if not self.for_inference:
                if sample.reference_annotation is not None:
                    # assume masked clm
                    tokenization_output = self.tokenizer(
                        sample.reference_annotation,
                        return_tensors="pt",
                        **(
                            {"truncation": True, "max_length": self.max_length}
                            if self.truncation
                            else {}
                        ),
                    )
                    elem_dict["labels"] = torch.cat(
                        (
                            torch.full_like(elem_dict["input_ids"], fill_value=-100),
                            tokenization_output["input_ids"].squeeze(),
                        ),
                        dim=0,
                    )
                    elem_dict["input_ids"] = torch.cat(
                        (
                            elem_dict["input_ids"],
                            tokenization_output["input_ids"].squeeze(),
                        ),
                        dim=0,
                    )
                    elem_dict["attention_mask"] = torch.cat(
                        (
                            elem_dict["attention_mask"],
                            tokenization_output["attention_mask"].squeeze(),
                        ),
                        dim=0,
                    )
                else:
                    # assume standard clm
                    elem_dict["labels"] = (
                        tokenization_output["input_ids"].squeeze().clone()
                    )

            yield elem_dict
