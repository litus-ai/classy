from typing import NamedTuple, Iterable, List, Union
import json

import logging

logger = logging.getLogger(__name__)


class SentencePairSample(NamedTuple):
    sentence1: str
    sentence2: str
    label: str


class SequenceSample(NamedTuple):
    sequence: str
    label: str


class TokensSample(NamedTuple):
    tokens: List[str]
    labels: List[str]


class FileReader:
    def read(self, path: str) -> Iterable[Union[SentencePairSample, SequenceSample, TokensSample]]:
        raise NotImplementedError


class SentencePairReader(FileReader):
    def read(self, path: str) -> Iterable[SentencePairSample]:
        raise NotImplementedError


class SequenceReader(FileReader):
    def read(self, path: str) -> Iterable[SequenceSample]:
        raise NotImplementedError


class TokensReader(FileReader):
    def read(self, path: str) -> Iterable[TokensSample]:
        raise NotImplementedError


class TSVSentencePairReader(SentencePairReader):
    def read(self, path: str) -> Iterable[SentencePairSample]:
        with open(path, "r") as f:
            for line in f:
                sentence1, sentence2, label = line.strip().split("\t")
                yield SentencePairSample(sentence1, sentence2, label)


class JSONLSentencePairReader(SentencePairReader):
    def read(self, path: str) -> Iterable[SentencePairSample]:
        with open(path, "r") as f:
            for line in f:
                yield SentencePairSample(**json.loads(line))


class TSVSequenceReader(SequenceReader):
    def read(self, path: str) -> Iterable[SequenceSample]:
        with open(path, "r") as f:
            for line in f:
                sentence, label = line.strip().split("\t")
                yield SequenceSample(sentence, label)


class JSONLSequenceReader(SequenceReader):
    def read(self, path: str) -> Iterable[SequenceSample]:
        with open(path, "r") as f:
            for line in f:
                yield SequenceSample(**json.loads(line))


class TSVTokensReader(TokensReader):
    def read(self, path: str) -> Iterable[TokensSample]:
        with open(path, "r") as f:
            for line in f:
                tokens, labels = line.strip().split("\t")
                yield TokensSample(tokens.split(" "), labels.split(" "))


class JSONLTokensReader(TokensReader):
    def read(self, path: str) -> Iterable[TokensSample]:
        with open(path, "r") as f:
            for line in f:
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


def get_reader(task_type: str, file_extension: str) -> FileReader:
    reader_identifier = (task_type, file_extension)
    if reader_identifier not in READERS_DICT:
        logger.info(f"No reader available for task {task_type} and extension {file_extension}.")
    return READERS_DICT[reader_identifier]()
