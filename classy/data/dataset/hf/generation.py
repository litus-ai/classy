import collections
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional

import torch
from omegaconf import DictConfig
from transformers import MBartTokenizerFast
from transformers.models.mbart.tokenization_mbart_fast import FAIRSEQ_LANGUAGE_CODES

from classy.data.data_drivers import ClassySample
from classy.data.dataset.base import batchify
from classy.data.dataset.hf.base import HFBaseDataset
from classy.utils.log import get_project_logger
from classy.utils.vocabulary import Vocabulary

logger = get_project_logger(__name__)


class HFGenerationBaseDataset(HFBaseDataset):
    @staticmethod
    def requires_vocab() -> bool:
        return False

    @staticmethod
    def fit_vocabulary(samples: Iterator[ClassySample]) -> Vocabulary:
        raise NotImplementedError

    @classmethod
    def adapt_dataset_from(cls, training_dataset: DictConfig, setting: str):
        dataset = super().adapt_dataset_from(training_dataset, setting)
        if setting == "prediction":
            dataset["teacher_forcing"] = False
        return dataset

    def __init__(self, teacher_forcing: Optional[bool] = True, **kwargs):
        self.teacher_forcing = teacher_forcing
        super().__init__(
            batching_fields=self.get_batching_fields(self.teacher_forcing), **kwargs
        )

    def get_batching_fields(self, teacher_forcing: bool):
        raise NotImplementedError

    def materialize_batches(
        self, dataset_elements: List[Dict[str, Any]]
    ) -> Generator[Dict[str, Any], None, None]:
        for group in self.group_elements_on_materializations(
            dataset_elements, teacher_forcing=self.teacher_forcing
        ):
            yield from super().materialize_batches(group)

    def group_elements_on_materializations(
        self, dataset_elements: List[Dict[str, Any]], teacher_forcing: bool
    ) -> List[List[Dict[str, Any]]]:
        return [dataset_elements]


class EncDecHFGenerationBaseDataset(HFGenerationBaseDataset):
    def get_batching_fields(self, teacher_forcing: bool) -> List[str]:
        return ["input_ids", "labels"] if teacher_forcing else ["input_ids"]


class DecHFGenerationBaseDataset(HFGenerationBaseDataset):
    def get_batching_fields(self, teacher_forcing: bool) -> List[str]:
        return ["input_ids"]

    def group_elements_on_materializations(
        self, dataset_elements: List[Dict[str, Any]], teacher_forcing: bool
    ) -> List[List[Dict[str, Any]]]:

        if teacher_forcing:
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

            if self.teacher_forcing:
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
            ), f"MBartHFGenerationDataset requires language specification"

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

            if self.teacher_forcing:
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
        self, dataset_elements: List[Dict[str, Any]], teacher_forcing: bool
    ) -> List[List[Dict[str, Any]]]:
        if teacher_forcing:
            return [dataset_elements]

        groups = collections.defaultdict(list)

        for de in dataset_elements:
            groups[de["samples"].target_language].append(de)

        return [group for group_len, group in groups.items()]


class T5HFGenerationDataset(EncDecHFGenerationBaseDataset):
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
        }

    def dataset_iterator_func(self):
        for sample in self.samples_iterator():

            assert (
                sample.source_language is None and sample.target_language is None
            ), f"T5HFGenerationDataset requires task/language specification to be set in sample.source_sequence"

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

            if self.teacher_forcing:
                if sample.reference_annotation is not None:
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

            elem_dict["samples"] = sample
            yield elem_dict


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
            if self.teacher_forcing:
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
