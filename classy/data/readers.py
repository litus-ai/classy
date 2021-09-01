from typing import NamedTuple, Iterable, List, Union, Optional
import json

import logging

from classy.utils.log import get_project_logger

logger = get_project_logger(__name__)


class SentencePairSample(NamedTuple):
    sentence1: str
    sentence2: str
    label: Optional[str] = None


class SequenceSample(NamedTuple):
    sequence: str
    label: Optional[str] = None


class TokensSample(NamedTuple):
    tokens: List[str]
    labels: Optional[List[str]] = None


class Reader:

    def read_from_path(self, path: str) -> Iterable[Union[SentencePairSample, SequenceSample, TokensSample]]:
        def r():
            with open(path) as f:
                for line in f:
                    yield line.strip()
        return self.read(r())

    def read(self, lines: Iterable[str]) -> Iterable[Union[SentencePairSample, SequenceSample, TokensSample]]:
        raise NotImplementedError


class SentencePairReader(Reader):
    def read(self, lines: Iterable[str]) -> Iterable[SentencePairSample]:
        raise NotImplementedError


class SequenceReader(Reader):
    def read(self, lines: Iterable[str]) -> Iterable[SequenceSample]:
        raise NotImplementedError


class TokensReader(Reader):
    def read(self, lines: Iterable[str]) -> Iterable[TokensSample]:
        raise NotImplementedError


class TSVSentencePairReader(SentencePairReader):
    def read(self, lines: Iterable[str]) -> Iterable[SentencePairSample]:
        for line in lines:
            parts = line.split('\t')
            assert len(parts) in [2, 3], f'TSVSentencePairReader expects 2 (s1, s2) or 3 (s1, s2, label) fields, but {len(parts)} were found'
            sentence1, sentence2 = parts[0], parts[1]
            label = parts[2] if len(parts) == 3 else None
            yield SentencePairSample(sentence1, sentence2, label)


class JSONLSentencePairReader(SentencePairReader):
    def read(self, lines: Iterable[str]) -> Iterable[SentencePairSample]:
        for line in lines:
            yield SentencePairSample(**json.loads(line))


class TSVSequenceReader(SequenceReader):
    def read(self, lines: Iterable[str]) -> Iterable[SequenceSample]:
        for line in lines:
            parts = line.split('\t')
            assert len(parts) in [1, 2], f'TSVSequenceReader expects 1 (sentence) or 3 (sentence, label) fields, but {len(parts)} were found at line {line}'
            sentence = parts[0]
            label = parts[1] if len(parts) == 2 else None
            yield SequenceSample(sentence, label)


class JSONLSequenceReader(SequenceReader):
    def read(self, lines: Iterable[str]) -> Iterable[SequenceSample]:
        for line in lines:
            yield SequenceSample(**json.loads(line))


class TSVTokensReader(TokensReader):
    def read(self, lines: Iterable[str]) -> Iterable[TokensSample]:
        for line in lines:
            parts = line.split('\t')
            assert len(parts) in [1, 2], f'TSVTokensReader expects 1 (tokens) or 3 (tokens, labels) fields, but {len(parts)} were found at line {line}'
            tokens, labels = parts[0], None
            if len(parts) == 2:
                labels = parts[2]
                assert len(tokens) == len(labels), f'Token Classification requires as many token as labels: found {len(tokens)} tokens != {len(labels)} labels at line {line}'
            tokens, labels = line.strip().split("\t")
            yield TokensSample(tokens.split(" "), labels.split(" "))


class JSONLTokensReader(TokensReader):
    def read(self, lines: Iterable[str]) -> Iterable[TokensSample]:
        for line in lines:
            sample = TokensSample(**json.loads(line))
            if sample.labels is not None:
                assert len(sample.tokens) == len(sample.labels), f'Token Classification requires as many token as labels: found {len(sample.tokens)} tokens != {len(sample.labels)} labels at line {line}'
            yield TokensSample(**json.loads(line))


# TASK TYPES
SEQUENCE = "sequence"
SENTENCE_PAIR = "sentence-pair"
TOKEN = "token"

# FILE EXTENSIONS
TSV = "tsv"
JSONL = "jsonl"

READERS_DICT = {
    (SEQUENCE, TSV): TSVSequenceReader,
    (SENTENCE_PAIR, TSV): TSVSentencePairReader,
    (TOKEN, TSV): TSVTokensReader,
    (SEQUENCE, JSONL): JSONLSequenceReader,
    (SENTENCE_PAIR, JSONL): JSONLSentencePairReader,
    (TOKEN, JSONL): JSONLTokensReader,
}


def get_reader(task_type: str, file_extension: str) -> Reader:
    reader_identifier = (task_type, file_extension)
    if reader_identifier not in READERS_DICT:
        logger.info(f"No reader available for task {task_type} and extension {file_extension}.")
    return READERS_DICT[reader_identifier]()
