import json

from classy.data.data_drivers import (
    GENERATION,
    JSONL,
    QA,
    SENTENCE_PAIR,
    SEQUENCE,
    TOKEN,
    ClassySample,
    GenerationSample,
    QASample,
    SentencePairSample,
    SequenceSample,
    TokensSample,
    get_data_driver,
)
from classy.pl_modules.mixins.task_serve import (
    GenerationTaskServeMixin,
    QATaskServeMixin,
    SentencePairTaskServeMixin,
    SequenceTaskServeMixin,
    TokenTaskServeMixin,
)
from classy.pl_modules.mixins.task_ui import (
    GenerationTaskUIMixin,
    QATaskUIMixin,
    SentencePairTaskUIMixin,
    SequenceTaskUIMixin,
    TokenTaskUIMixin,
)


class TaskMixin:
    def read_input_from_bash(
        self,
    ) -> ClassySample:
        raise NotImplementedError

    @property
    def task(self) -> str:
        raise NotImplementedError


class SequenceTask(SequenceTaskServeMixin, SequenceTaskUIMixin, TaskMixin):

    __data_driver = get_data_driver(SEQUENCE, JSONL)

    def read_input_from_bash(self) -> SequenceSample:
        sequence = input("Enter sequence text: ").strip()
        sample = json.dumps({"sequence": sequence})
        return next(self.__data_driver.read([sample]))

    @property
    def task(self) -> str:
        return SEQUENCE


class TokensTask(TokenTaskServeMixin, TokenTaskUIMixin, TaskMixin):
    __data_driver = get_data_driver(TOKEN, JSONL)

    def read_input_from_bash(self) -> TokensSample:
        tokens = input("Enter space-separated tokens: ").strip()
        sample = json.dumps({"tokens": tokens.split(" ")})
        return next(self.__data_driver.read([sample]))

    @property
    def task(self) -> str:
        return TOKEN


class GenerationTask(GenerationTaskServeMixin, GenerationTaskUIMixin, TaskMixin):
    __data_driver = get_data_driver(GENERATION, JSONL)

    def read_input_from_bash(self) -> GenerationSample:
        source_sequence = input("Enter source sequence text: ").strip()
        source_language = (
            input("Enter source language (leave empty to set it to None): ").strip()
            or None
        )
        target_language = (
            input("Enter target language (leave empty to set it to None): ").strip()
            or None
        )
        sample = json.dumps(
            dict(
                source_sequence=source_sequence,
                source_language=source_language,
                target_language=target_language,
            )
        )
        return next(self.__data_driver.read([sample]))

    @property
    def task(self) -> str:
        return GENERATION


class SentencePairTask(SentencePairTaskServeMixin, SentencePairTaskUIMixin, TaskMixin):
    __data_driver = get_data_driver(SENTENCE_PAIR, JSONL)

    def read_input_from_bash(self) -> SentencePairSample:
        sentence1 = input("Enter first sentence: ").strip()
        sentence2 = input("Enter second sentence: ").strip()
        sample = json.dumps({"sentence1": sentence1, "sentence2": sentence2})
        return next(self.__data_driver.read([sample]))

    @property
    def task(self) -> str:
        return SENTENCE_PAIR


class QATask(QATaskServeMixin, QATaskUIMixin, TaskMixin):
    __data_driver = get_data_driver(QA, JSONL)

    def read_input_from_bash(self) -> QASample:
        question = input("Enter question: ").strip()
        context = input("Enter context: ").strip()
        sample = json.dumps({"question": question, "context": context})
        return next(self.__data_driver.read([sample]))

    @property
    def task(self) -> str:
        return QA
