from functools import lru_cache
from typing import List, NamedTuple, Optional, Dict, Tuple, Union

import torch
from tokenizers import AddedToken

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from classy.utils.commons import flatten


class TokenizationOutput(NamedTuple):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: Optional[torch.Tensor]
    definitions_offsets: Dict[str, Tuple[int, int]]
    relative_positions: Optional[torch.Tensor] = None


# utilities to extract the gold definition indices in the input sequence
def extract_gold_indices(
    definition: str, definitions_sequence: str, offsets_mapping: List[Tuple[int, int]]
) -> Tuple[int, int]:
    gold_start_offset = definitions_sequence.index(definition)
    gold_end_offset = gold_start_offset + len(definition)

    offset_start2bpe_idx = {}
    offset_end2bpe_idx = {}
    for i, (off_start, off_end) in enumerate(offsets_mapping):

        if off_start == off_end:
            continue  # specials

        offset_start2bpe_idx[off_start] = i
        offset_end2bpe_idx[off_end] = i

    start_bpe_idx = offset_start2bpe_idx[gold_start_offset]
    end_bpe_idx = offset_end2bpe_idx[gold_end_offset]

    assert start_bpe_idx < end_bpe_idx

    return start_bpe_idx, end_bpe_idx


class ConsecTokenizer:

    _shared_state = {}

    def __init__(
        self,
        transformer_model: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast],
        target_marker: Tuple[str, str],
        context_definitions_token: str,
        context_markers: Dict,
        add_prefix_space: bool,
    ):
        if type(transformer_model) == str:
            if f"tokenizer-{transformer_model}" not in self._shared_state:
                self._shared_state[f"tokenizer-{transformer_model}"] = AutoTokenizer.from_pretrained(transformer_model)
            self.tokenizer = self._shared_state[f"tokenizer-{transformer_model}"]
        else:
            self.tokenizer = transformer_model
        self.target_marker = target_marker
        self.context_definitions_token = context_definitions_token
        self.context_markers = [
            (context_markers["pattern"][0].replace("#I#", f"{i}"), context_markers["pattern"][1].replace("#I#", f"{i}"))
            for i in range(context_markers["number"])
        ]
        assert (
            len(set(self.context_markers)) == context_markers["number"]
        ), f"Error in given pattern: number of unique created patterns != specified number"

        additional_special_tokens = [
            *[AddedToken(t, single_word=True, lstrip=True) for t in self.target_marker],
            *[AddedToken(t, single_word=True, lstrip=True) for p in self.context_markers for t in p],
            AddedToken(context_definitions_token, single_word=True, lstrip=True),
        ]
        self.tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
        self.n_added_special_tokens = len(additional_special_tokens)
        self.add_prefix_space = add_prefix_space

    def mark_token(self, token: str, marker: Tuple[str, str]) -> str:
        bom, eom = marker
        if self.add_prefix_space:
            return f"{bom} {token} {eom}"
        else:
            return f"{bom}{token}{eom}"

    def tokenize(
        self,
        sentence: List[str],  # tokens can be multiword
        instance_idx: int,
        instance_possible_definitions: List[str],
        context_definitions2positions: List[Tuple[str, int]],
    ) -> TokenizationOutput:
        raise NotImplementedError

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def model_max_length(self) -> int:
        return self.tokenizer.model_max_length


class DeBERTaTokenizer(ConsecTokenizer):
    def __init__(
        self,
        transformer_model: str,
        target_marker: Tuple[str, str],
        context_definitions_token: str,
        context_markers: Dict,
        add_prefix_space: bool,
        optimize_relative_positions: bool = True,
        enforce_symmetry: bool = True,
    ):
        super().__init__(transformer_model, target_marker, context_definitions_token, context_markers, add_prefix_space)
        self.optimize_relative_positions = optimize_relative_positions
        self.enforce_symmetry = enforce_symmetry

    def tokenize(
        self,
        sentence: List[str],  # tokens can be multiword
        instance_idx: int,
        instance_possible_definitions: List[str],
        context_definitions2positions: List[Tuple[str, int]],
        **kwargs,
    ) -> TokenizationOutput:
        if self.optimize_relative_positions:
            return self.power_tokenize(
                sentence, instance_idx, instance_possible_definitions, context_definitions2positions
            )
        else:
            return self.plain_tokenize(
                " ".join(sentence),
                instance_possible_definitions,
                [x[0] for x in context_definitions2positions],
                **kwargs,
            )

    def deberta_tokenize(self, text: str) -> List[int]:
        return self.tokenizer(text, return_attention_mask=False, return_token_type_ids=False, add_special_tokens=True,)[
            "input_ids"
        ][1:-1]

    def plain_tokenize(
        self,
        sentence: Union[str, List[int]],
        instance_possible_definitions: Union[List[str], List[Tuple[str, List[int]]]],
        context_definitions: Union[List[str], List[Tuple[str, List[int]]]],
        use_specials: bool = True,
    ) -> TokenizationOutput:
        if type(sentence) != list:
            sentence = f" {sentence}"
            sentence_input_ids = self.deberta_tokenize(sentence)
        else:
            sentence_input_ids = sentence

        final_input_ids = [self.tokenizer.cls_token_id] + sentence_input_ids + [self.tokenizer.sep_token_id]
        token_type_ids = [0] * len(final_input_ids)

        definitions_offsets = dict()
        for definition in instance_possible_definitions:
            if type(definition) == tuple:
                definition, definition_ids = definition
            else:
                definition_ids = self.deberta_tokenize(f" {definition}")

            definitions_offsets[definition] = len(final_input_ids), len(final_input_ids) + len(definition_ids)

            final_input_ids += definition_ids
            token_type_ids += [1] * len(definition_ids)

        # "context" definitions token
        if self.context_definitions_token is not None and use_specials:
            final_input_ids += self.deberta_tokenize(self.context_definitions_token)
            token_type_ids.append(1)
        else:
            if use_specials:
                assert len(context_definitions) == 0

        for definition in context_definitions:
            if type(definition) == tuple:
                definition, definition_ids = definition
            else:
                definition_ids = self.deberta_tokenize(f" {definition}")

            final_input_ids += definition_ids
            token_type_ids += [1] * len(definition_ids)

        # last [SEP] token
        final_input_ids.append(self.tokenizer.sep_token_id)
        token_type_ids.append(1)

        final_input_ids = torch.tensor(final_input_ids, dtype=torch.long)
        attention_mask = torch.ones_like(final_input_ids)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)

        return TokenizationOutput(final_input_ids, attention_mask, token_type_ids, definitions_offsets)

    @lru_cache(maxsize=10_000)
    def split_tokenize(self, word: str):
        return self.deberta_tokenize(f" {word.strip()}")

    @lru_cache(maxsize=10_000)
    def _tokenize_sentence_list(self, sentence: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        sentence = sentence.split("\t")
        sentence_input_ids = [self.tokenizer.cls_token_id]
        token_offsets = []

        for token in sentence:
            token_ids = self.split_tokenize(f" {token.strip()}")
            token_offsets.append((len(sentence_input_ids), len(sentence_input_ids) + len(token_ids)))
            sentence_input_ids += token_ids

        sentence_input_ids.append(self.tokenizer.sep_token_id)

        return sentence_input_ids, token_offsets

    @staticmethod
    def mirror_arange(size: int, zero_pos: int):
        return torch.cat([torch.flip(torch.arange(0, zero_pos + 1), dims=(0,)), -torch.arange(1, size - zero_pos)])

    def power_tokenize(
        self,
        sentence: List[str],  # tokens can be multiword
        instance_idx: int,
        instance_possible_definitions: List[str],
        context_definitions2positions: List[Tuple[str, int]],
    ) -> TokenizationOutput:

        sentence_input_ids, token_offsets = self._tokenize_sentence_list("\t".join(sentence))

        instance_possible_definitions_ids = [self.deberta_tokenize(f" {ipd}") for ipd in instance_possible_definitions]
        context_definitions_ids = [self.deberta_tokenize(f" {cd}") for cd, _ in context_definitions2positions]

        total_input_ids = (
            len(sentence_input_ids)
            + len(flatten(instance_possible_definitions_ids))
            + len(flatten(context_definitions_ids))
        )

        relative_positions = torch.zeros((total_input_ids + 1, total_input_ids + 1), dtype=torch.long)

        # filling the sentences relative positions
        for id_idx in range(len(sentence_input_ids)):
            relative_positions[id_idx, :-1] = torch.cat(
                [self.mirror_arange(len(sentence_input_ids), id_idx)]
                + [
                    -torch.arange(len(sentence_input_ids) - id_idx, len(sentence_input_ids) + len(def_ids) - id_idx)
                    for def_ids in instance_possible_definitions_ids + context_definitions_ids
                ]
            )

        # flipping the matrix
        relative_positions.T[: len(sentence_input_ids)] = -relative_positions[: len(sentence_input_ids)]

        curr_offset = len(sentence_input_ids)

        # computing the relative positions for the possible definitions
        definitions2positions = [(instance_idx, ipdid) for ipdid in instance_possible_definitions_ids]
        definitions2positions += [
            (rel_idx, cdid) for cdid, (_, rel_idx) in zip(context_definitions_ids, context_definitions2positions)
        ]
        for def_num in range(len(definitions2positions)):

            rel_token_idx, def_ids = definitions2positions[def_num]
            possible_def_token_pos = list(range(len(def_ids)))
            instance_start_pos, instance_end_pos = token_offsets[rel_token_idx]

            for token_pos in possible_def_token_pos:

                for off_idx, inst_token_pos in enumerate(range(instance_start_pos, instance_end_pos)):
                    relative_positions[curr_offset + token_pos, inst_token_pos] = (
                        token_pos + (instance_end_pos - instance_start_pos) - off_idx
                    )
                    if self.enforce_symmetry or rel_token_idx != instance_idx:
                        relative_positions[inst_token_pos, curr_offset + token_pos] = -relative_positions[
                            curr_offset + token_pos, inst_token_pos
                        ]

                relative_positions[curr_offset + token_pos, len(sentence_input_ids) : -1] = self.mirror_arange(
                    total_input_ids - len(sentence_input_ids), curr_offset + token_pos - len(sentence_input_ids)
                )

            curr_offset += len(possible_def_token_pos)

        last_token_positions = torch.min(relative_positions, dim=-1)[0] - 1
        relative_positions[-1] = -last_token_positions
        relative_positions.T[-1] = last_token_positions
        relative_positions[-1, -1] = 0

        relative_positions[relative_positions == -0] = 0

        tokenization_output = self.plain_tokenize(
            sentence_input_ids[1:-1],
            [(x, y) for x, y in zip(instance_possible_definitions, instance_possible_definitions_ids)],
            [(x, y) for x, y in zip([x[0] for x in context_definitions2positions], context_definitions_ids)],
            use_specials=False,
        )

        assert tokenization_output.input_ids.shape[0] == relative_positions.shape[0]

        return TokenizationOutput(
            tokenization_output.input_ids,
            tokenization_output.attention_mask,
            tokenization_output.token_type_ids,
            tokenization_output.definitions_offsets,
            relative_positions,
        )

    @property
    def model_max_length(self) -> int:
        return 24_528  # from paper: deberta-large


class MBartTokenizer(ConsecTokenizer):
    def __init__(
        self,
        transformer_model: str,
        target_marker: Tuple[str, str],
        context_definitions_token: str,
        context_markers: Dict,
        add_prefix_space: bool,
        source_language: str = "en_XX",
        target_language: str = "en_EN",
    ):
        tokenizer = AutoTokenizer.from_pretrained(transformer_model, src_lang=source_language, tgt_lang=target_language)
        self.source_language = source_language
        super().__init__(tokenizer, target_marker, context_definitions_token, context_markers, add_prefix_space)

    def mbart_tokenize(self, text: str) -> List[int]:
        tokenization_out = self.tokenizer(
            text,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )["input_ids"]
        return tokenization_out[:-2]

    def tokenize(
        self,
        sentence: Union[str, List[int]],
        instance_idx: int,
        instance_possible_definitions: List[str],
        context_definitions2positions: List[Tuple[str, int]],
    ) -> TokenizationOutput:
        use_specials = False
        context_definitions = [x[0] for x in context_definitions2positions]

        sentence = f" {' '.join(sentence)}"
        sentence_input_ids = self.mbart_tokenize(sentence)

        final_input_ids = sentence_input_ids

        definitions_offsets = dict()
        for definition in instance_possible_definitions:
            if type(definition) == tuple:
                definition, definition_ids = definition
            else:
                definition_ids = self.mbart_tokenize(f" {definition}")

            definitions_offsets[definition] = len(final_input_ids), len(final_input_ids) + len(definition_ids)

            final_input_ids += definition_ids

        # "context" definitions token
        if self.context_definitions_token is not None and use_specials:
            final_input_ids += self.mbart_tokenize(self.context_definitions_token)
        else:
            if use_specials:
                assert len(context_definitions) == 0

        for definition in context_definitions:
            if type(definition) == tuple:
                definition, definition_ids = definition
            else:
                definition_ids = self.mbart_tokenize(f" {definition}")

            final_input_ids += definition_ids

        # last [SEP] token
        final_input_ids.append(self.tokenizer.sep_token_id)

        # lang id
        final_input_ids += self.mbart_tokenize(self.source_language)

        final_input_ids = torch.tensor(final_input_ids, dtype=torch.long)
        attention_mask = torch.ones_like(final_input_ids)

        return TokenizationOutput(final_input_ids, attention_mask, None, definitions_offsets)
