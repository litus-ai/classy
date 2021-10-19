import json
from typing import Union

from classy.data.data_drivers import (
    SENTENCE_PAIR,
    QA,
    TOKEN,
    SEQUENCE,
    SentencePairSample,
    SequenceSample,
    TokensSample,
    QASample,
    get_data_driver,
    JSONL,
    GENERATION,
    GenerationSample,
)
from classy.pl_modules.mixins.task_ui import (
    TaskUIMixin,
    SequenceTaskUIMixin,
    TokenTaskUIMixin,
    SentencePairTaskUIMixin,
    QATaskUIMixin,
    GenerationTaskUIMixin,
)


class TaskMixin(TaskUIMixin):
    def read_input_from_bash(self) -> Union[SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample]:
        raise NotImplementedError

    @property
    def task(self) -> str:
        raise NotImplementedError


class SequenceTask(SequenceTaskUIMixin, TaskMixin):

    __data_driver = get_data_driver(SEQUENCE, JSONL)

    def read_input_from_bash(self) -> SequenceSample:
        sequence = input("Enter sequence text: ")
        sample = json.dumps({"sequence": sequence})
        return next(self.__data_driver.read([sample]))

    @property
    def task(self) -> str:
        return SEQUENCE


class TokensTask(TokenTaskUIMixin, TaskMixin):
    __data_driver = get_data_driver(TOKEN, JSONL)

    def read_input_from_bash(self) -> TokensSample:
        tokens = input("Enter space-separated tokens: ")
        sample = json.dumps({"tokens": tokens.split(" ")})
        return next(self.__data_driver.read([sample]))

    @property
    def task(self) -> str:
        return TOKEN


class GenerationTask(GenerationTaskUIMixin, TaskMixin):
    __data_driver = get_data_driver(GENERATION, JSONL)

    def read_input_from_bash(self) -> GenerationSample:
        source_sequence = input("Enter source sequence text: ").strip()
        source_language = input("Enter source language (leave empty to set it to None): ").strip() or None
        target_language = input("Enter target language (leave empty to set it to None): ").strip() or None
        sample = json.dumps(
            dict(source_sequence=source_sequence, source_language=source_language, target_language=target_language)
        )
        return next(self.__data_driver.read([sample]))

    @property
    def task(self) -> str:
        return GENERATION


class SentencePairTask(SentencePairTaskUIMixin, TaskMixin):
    __data_driver = get_data_driver(SENTENCE_PAIR, JSONL)

    def read_input_from_bash(self) -> SentencePairSample:
        sentence1 = input("Enter first sentence: ")
        sentence2 = input("Enter second sentence: ")
        sample = json.dumps({"sentence1": sentence1, "sentence2": sentence2})
        return next(self.__data_driver.read([sample]))

    @property
    def task(self) -> str:
        return SENTENCE_PAIR


class QATask(QATaskUIMixin, TaskMixin):
    __data_driver = get_data_driver(QA, JSONL)

    def read_input_from_bash(self) -> QASample:
        question = input("Enter question: ")
        context = input("Enter context: ")
        sample = json.dumps({"question": question, "context": context})
        return next(self.__data_driver.read([sample]))

    @property
    def task(self) -> str:
        return QA
