from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch

from classy.data.data_drivers import (
    QASample,
    SentencePairSample,
    SequenceSample,
    TokensSample,
)
from classy.data.dataset.base import batchify
from classy.data.dataset.hf.base import HFBaseDataset
from classy.utils.log import get_project_logger
from classy.utils.vocabulary import Vocabulary

logger = get_project_logger(__name__)


class HFSequenceDataset(HFBaseDataset):
    @staticmethod
    def fit_vocabulary(samples: Iterator[SequenceSample]) -> Vocabulary:
        return Vocabulary.from_samples(
            [{"labels": sample.reference_annotation} for sample in samples]
        )

    def init_fields_batcher(self) -> Dict:
        return {
            "input_ids": lambda lst: batchify(
                lst, padding_value=self.tokenizer.pad_token_id
            ),
            "attention_mask": lambda lst: batchify(lst, padding_value=0),
            "labels": lambda lst: torch.tensor(lst, dtype=torch.long).squeeze(-1),
            "samples": None,
        }

    def dataset_iterator_func(self) -> Iterable[Dict[str, Any]]:

        for sequence_sample in self.samples_iterator():
            input_ids = self.tokenizer(sequence_sample.sequence, return_tensors="pt")[
                "input_ids"
            ][0]
            elem_dict = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
            if sequence_sample.reference_annotation is not None:
                elem_dict["labels"] = [
                    self.vocabulary.get_idx(
                        k="labels", elem=sequence_sample.reference_annotation
                    )
                ]
            elem_dict["samples"] = sequence_sample
            yield elem_dict


class HFTokenDataset(HFBaseDataset):
    @staticmethod
    def fit_vocabulary(samples: Iterator[TokensSample]) -> Vocabulary:
        return Vocabulary.from_samples(
            [
                {"labels": label}
                for sample in samples
                for label in sample.reference_annotation
            ]
        )

    def init_fields_batcher(self) -> Dict:
        return {
            "input_ids": lambda lst: batchify(
                lst, padding_value=self.tokenizer.pad_token_id
            ),
            "attention_mask": lambda lst: batchify(lst, padding_value=0),
            "labels": lambda lst: batchify(
                lst, padding_value=-100
            ),  # -100 == cross entropy ignore index
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
            if token_sample.reference_annotation is not None:
                elem_dict["labels"] = torch.tensor(
                    [
                        self.vocabulary.get_idx(
                            k="labels", elem=token_sample.reference_annotation[idx]
                        )
                        for idx in token_sample.target
                    ]
                )

            elem_dict["samples"] = token_sample
            yield elem_dict

    def tokenize(
        self, tokens: List[str]
    ) -> Optional[Tuple[torch.Tensor, List[Tuple[int, int]]]]:
        tok_encoding = self.tokenizer.encode_plus(
            tokens, return_tensors="pt", is_split_into_words=True
        )
        try:
            return tok_encoding.input_ids.squeeze(0), [
                tuple(tok_encoding.word_to_tokens(wi)) for wi in range(len(tokens))
            ]
        except TypeError:
            logger.warning(f"Tokenization failed for tokens: {' | '.join(tokens)}")
            return None


class HFSentencePairDataset(HFSequenceDataset):
    def init_fields_batcher(self) -> Dict:
        fields_batcher = super(HFSentencePairDataset, self).init_fields_batcher()
        fields_batcher["token_type_ids"] = lambda lst: batchify(lst, padding_value=0)
        return fields_batcher

    def dataset_iterator_func(self) -> Iterable[Dict[str, Any]]:

        for sequence_sample in self.samples_iterator():
            sequence_sample: SentencePairSample
            tokenization_output = self.tokenizer(
                sequence_sample.sentence1,
                sequence_sample.sentence2,
                return_tensors="pt",
            )

            elem_dict = {
                "input_ids": tokenization_output["input_ids"].squeeze(),
                "attention_mask": tokenization_output["attention_mask"].squeeze(),
            }

            if "token_type_ids" in tokenization_output:
                elem_dict["token_type_ids"] = tokenization_output[
                    "token_type_ids"
                ].squeeze()

            if sequence_sample.reference_annotation is not None:
                elem_dict["labels"] = [
                    self.vocabulary.get_idx(
                        k="labels", elem=sequence_sample.reference_annotation
                    )
                ]

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
                "context_mask": torch.tensor(
                    [p == 1 for p in tokenization_output.sequence_ids()]
                ),
            }

            if "token_type_ids" in tokenization_output:
                elem_dict["token_type_ids"] = tokenization_output[
                    "token_type_ids"
                ].squeeze(0)

            # use token2chars to build the mapping char2token
            # we should be using tokenization_output.char_to_token but there
            # seems to be some bug around it with some tokenizers (e.g. BartTokenizer)
            # t("Question", "X Y").char_to_token(1, sequence_id=1) returns None
            # (first paper to char_to_token is 1 to account for added prefix space)
            char2token = {}
            first = True
            for _t_idx, (m, cp) in enumerate(
                zip(
                    elem_dict["context_mask"].tolist(),
                    elem_dict["token2chars"].tolist(),
                )
            ):
                if m:

                    # postprocess token2chars
                    # some tokenizers (microsoft/deberta-base) include in the token-offsets also the white space
                    # e.g. 'In Italy' => ' Italy' => (2, 8)
                    # set position to first non-white space
                    while (
                        elem_dict["token2chars"][_t_idx][0]
                        < elem_dict["token2chars"][_t_idx][1]
                        and qa_sample.context[
                            elem_dict["token2chars"][_t_idx][0].item()
                        ]
                        == " "
                    ):
                        elem_dict["token2chars"][_t_idx][0] += 1

                    # add prefix space seems to be bugged on some tokenizers
                    if first:
                        first = False
                        if cp[0] != 0 and qa_sample.context[cp[0] - 1] != " ":
                            # this is needed to cope with tokenizers such as bart
                            # where t("Question", "X Y").token2chars[<bpe of X>] = (1, 1)
                            elem_dict["token2chars"][_t_idx][0] -= 1
                            cp = (cp[0] - 1, cp[1])
                    if cp[0] == cp[1]:
                        # this happens on some tokenizers when multiple spaces are present
                        assert (
                            qa_sample.context[cp[0] - 1] == " "
                        ), f"Token {_t_idx} found to occur at char span ({cp[0]}, {cp[1]}), which is impossible"
                    for c in range(*cp):
                        char2token[c] = _t_idx

            if qa_sample.reference_annotation is not None:
                char_start, char_end = qa_sample.reference_annotation
                elem_dict["start_position"] = char2token[char_start]
                elem_dict["end_position"] = char2token[char_end - 1]
                if (
                    elem_dict["start_position"] is None
                    or elem_dict["end_position"] is None
                ):
                    continue

            elem_dict["samples"] = qa_sample

            yield elem_dict

    def init_fields_batcher(self) -> Dict:
        return {
            "input_ids": lambda lst: batchify(
                lst, padding_value=self.tokenizer.pad_token_id
            ),
            "attention_mask": lambda lst: batchify(lst, padding_value=0),
            "token_type_ids": lambda lst: batchify(lst, padding_value=0),
            "context_mask": lambda lst: batchify(lst, padding_value=0),
            "token2chars": None,
            "start_position": lambda lst: torch.tensor(lst, dtype=torch.long),
            "end_position": lambda lst: torch.tensor(lst, dtype=torch.long),
            "samples": None,
        }
