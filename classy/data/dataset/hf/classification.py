from typing import Optional, Iterable, Dict, Any, Tuple, Iterator, List

import torch

from classy.data.data_drivers import SequenceSample, TokensSample, SentencePairSample, QASample
from classy.data.dataset.base import batchify
from classy.data.dataset.hf.base import HFBaseDataset
from classy.utils.log import get_project_logger
from classy.utils.vocabulary import Vocabulary

logger = get_project_logger(__name__)


class HFSequenceDataset(HFBaseDataset):
    @staticmethod
    def fit_vocabulary(samples: Iterator[SequenceSample]) -> Vocabulary:
        return Vocabulary.from_samples([{"labels": sample.label} for sample in samples])

    def _init_fields_batcher(self) -> None:
        self.fields_batcher = {
            "input_ids": lambda lst: batchify(lst, padding_value=self.tokenizer.pad_token_id),
            "attention_mask": lambda lst: batchify(lst, padding_value=0),
            "labels": lambda lst: torch.tensor(lst, dtype=torch.long).squeeze(-1),
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


class HFTokenDataset(HFBaseDataset):
    @staticmethod
    def fit_vocabulary(samples: Iterator[TokensSample]) -> Vocabulary:
        return Vocabulary.from_samples([{"labels": label} for sample in samples for label in sample.labels])

    def _init_fields_batcher(self) -> None:
        self.fields_batcher = {
            "input_ids": lambda lst: batchify(lst, padding_value=self.tokenizer.pad_token_id),
            "attention_mask": lambda lst: batchify(lst, padding_value=0),
            "labels": lambda lst: batchify(lst, padding_value=-100),  # -100 == cross entropy ignore index
            "samples": None,
            "token_offsets": None,
        }

    def dataset_iterator_func(self) -> Iterable[Dict[str, Any]]:

        for token_sample in self.samples_iterator():
            input_ids, token_offsets = self.tokenize(token_sample.tokens)
            elem_dict = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "token_offsets": token_offsets,
            }
            if token_sample.labels is not None:
                elem_dict["labels"] = torch.tensor(
                    [self.vocabulary.get_idx(k="labels", elem=token_sample.labels[idx]) for idx in token_sample.target]
                )

            elem_dict["samples"] = token_sample
            yield elem_dict

    def tokenize(self, tokens: List[str]) -> Optional[Tuple[torch.Tensor, List[Tuple[int, int]]]]:
        tok_encoding = self.tokenizer.encode_plus(tokens, return_tensors="pt", is_split_into_words=True)
        try:
            return tok_encoding.input_ids.squeeze(0), [
                tuple(tok_encoding.word_to_tokens(wi)) for wi in range(len(tokens))
            ]
        except TypeError:
            logger.warning(f"Tokenization failed for tokens: {' | '.join(tokens)}")
            return None


class HFSentencePairDataset(HFSequenceDataset):
    def _init_fields_batcher(self) -> None:
        super(HFSentencePairDataset, self)._init_fields_batcher()
        self.fields_batcher["token_type_ids"] = lambda lst: batchify(lst, padding_value=0)

    def dataset_iterator_func(self) -> Iterable[Dict[str, Any]]:

        for sequence_sample in self.samples_iterator():
            sequence_sample: SentencePairSample
            tokenization_output = self.tokenizer(
                sequence_sample.sentence1, sequence_sample.sentence2, return_tensors="pt"
            )

            elem_dict = {
                "input_ids": tokenization_output["input_ids"].squeeze(),
                "attention_mask": tokenization_output["attention_mask"].squeeze(),
            }

            if "token_type_ids" in tokenization_output:
                elem_dict["token_type_ids"] = tokenization_output["token_type_ids"].squeeze()

            if sequence_sample.label is not None:
                elem_dict["labels"] = [self.vocabulary.get_idx(k="labels", elem=sequence_sample.label)]

            elem_dict["samples"] = sequence_sample
            yield elem_dict


class HFQADataset(HFBaseDataset):
    @staticmethod
    def requires_vocab() -> bool:
        return False

    @staticmethod
    def fit_vocabulary(samples: Iterator[QASample]) -> Vocabulary:
        raise NotImplementedError

    def dataset_iterator_func(self) -> Iterable[Dict[str, Any]]:

        for qa_sample in self.samples_iterator():
            qa_sample: QASample

            tokenization_output = self.tokenizer(
                qa_sample.question,
                qa_sample.context,
                return_offsets_mapping=True,
                return_tensors="pt",
            )

            elem_dict = {
                "input_ids": tokenization_output["input_ids"].squeeze(0),
                "attention_mask": tokenization_output["attention_mask"].squeeze(0),
                "token2chars": tokenization_output["offset_mapping"].squeeze(0),
                "context_mask": torch.tensor([p == 1 for p in tokenization_output.sequence_ids()]),
            }

            if "token_type_ids" in tokenization_output:
                elem_dict["token_type_ids"] = tokenization_output["token_type_ids"].squeeze(0)

            if qa_sample.char_start is not None and qa_sample.char_end is not None:
                elem_dict["start_position"] = tokenization_output.char_to_token(qa_sample.char_start, sequence_index=1)
                elem_dict["end_position"] = tokenization_output.char_to_token(qa_sample.char_end - 1, sequence_index=1)
                if elem_dict["start_position"] is None or elem_dict["end_position"] is None:
                    continue

            elem_dict["samples"] = qa_sample

            yield elem_dict

    def _init_fields_batcher(self) -> None:
        self.fields_batcher = {
            "input_ids": lambda lst: batchify(lst, padding_value=self.tokenizer.pad_token_id),
            "attention_mask": lambda lst: batchify(lst, padding_value=0),
            "token_type_ids": lambda lst: batchify(lst, padding_value=0),
            "context_mask": lambda lst: batchify(lst, padding_value=0),
            "token2chars": None,
            "start_position": lambda lst: torch.tensor(lst, dtype=torch.long),
            "end_position": lambda lst: torch.tensor(lst, dtype=torch.long),
            "samples": None,
        }
