import functools
import json
from typing import List, Union, Optional, Iterator, Generator, Tuple, Dict, Any

from classy.utils.log import get_project_logger

logger = get_project_logger(__name__)


class ClassyStruct:
    def __init__(self, **kwargs):
        super().__setattr__("_d", {})
        self._d = kwargs

    def __getattr__(self, item):
        if item.startswith('__') and item.startswith('__'):
            # this is likely some python library-specific variable (such as __deepcopy__ for copy)
            # better follow standard behavior here
            raise AttributeError(item)
        elif item in self._d:
            return self._d[item]
        else:
            return None

    def __setattr__(self, key, value):
        if key in self._d:
            self._d[key] = value
        else:
            super().__setattr__(key, value)

    def get_additional_attributes(self) -> Dict[Any, Any]:
        return self._d

    def get_current_classification(self) -> Optional[Union[str, List[str], Tuple[int, int]]]:
        raise NotImplementedError

    def update_classification(self, classification_result: Union[str, List[str], Tuple[int, int]]):
        raise NotImplementedError

    def pretty_print(self, classification_result: Optional[Union[str, List[str], Tuple[int, int]]] = None) -> str:
        raise NotImplementedError


class SentencePairSample(ClassyStruct):
    def __init__(self, sentence1: str, sentence2: str, label: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.label = label

    def get_current_classification(self) -> Optional[str]:
        return self.label

    def update_classification(self, classification_result: str):
        self.label = classification_result

    def pretty_print(self, classification_result: Optional[str] = None) -> str:
        parts = [f"# sentence1: {self.sentence1}", f"# sentence2: {self.sentence2}"]
        if self.label is not None:
            parts.append(f"\t# label: {self.label}")
        if classification_result is not None:
            parts.append(f"\t# classification_result: {classification_result}")
        return "\n".join(parts)


class SequenceSample(ClassyStruct):
    def __init__(self, sequence: str, label: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.sequence = sequence
        self.label = label

    def get_current_classification(self) -> Optional[str]:
        return self.label

    def update_classification(self, classification_result: str):
        self.label = classification_result

    def pretty_print(self, classification_result: Optional[str] = None) -> str:
        parts = [f"# sequence: {self.sequence}"]
        if self.label is not None:
            parts.append(f"\t# label: {self.label}")
        if classification_result is not None:
            parts.append(f"\t# classification_result: {classification_result}")
        return "\n".join(parts)


class TokensSample(ClassyStruct):
    def __init__(
        self, tokens: List[str], labels: Optional[List[str]] = None, target: Optional[List[int]] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.tokens = tokens
        self.labels = labels
        self.target = target if target is not None else list(range(len(tokens)))
        self.was_target_provided = target is not None

    def get_current_classification(self) -> Optional[List[str]]:
        return self.labels

    def update_classification(self, classification_result: List[str]):
        self.labels = classification_result

    def pretty_print(self, classification_result: Optional[List[str]] = None) -> str:
        parts = [f'# tokens: {" ".join(self.tokens)}']
        if self.labels is not None:
            parts.append(f'\t# labels: {" ".join(self.labels)}')
        if classification_result is not None:
            parts.append(f'\t# classification_result: {" ".join(classification_result)}')
        return "\n".join(parts)


class QASample(ClassyStruct):
    def __init__(
        self,
        context: str,
        question: str,
        answer_start: Optional[int] = None,
        answer_end: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.context = context
        self.question = question
        self.char_start = answer_start
        self.char_end = answer_end

    def get_current_classification(self) -> Optional[Tuple[int, int]]:
        return self.char_start, self.char_end

    def update_classification(self, classification_result: Tuple[int, int]):
        self.char_start, self.char_end = classification_result

    def pretty_print(self, classification_result: Optional[Tuple[int, int]] = None) -> str:
        parts = [
            f"# context: {self.context}",
            f"# question: {self.question}",
        ]

        if self.char_start is not None and self.char_end is not None:
            parts += [
                "### Gold positions ###",
                f"# answer start: {self.char_start}, answer end: {self.char_end}",
                f"# answer: {self.context[self.char_start:self.char_end]}",
            ]

        if classification_result is not None:
            classification_start, classification_end = classification_result
            parts += [
                "### Predicted positions ###",
                f"# answer start: {classification_start}, answer end: {classification_end}",
                f"# answer: {self.context[classification_start:classification_end]}",
            ]

        return "\n".join(parts)


class GenerationSample(ClassyStruct):
    def __init__(
        self,
        source_sequence: str,
        source_language: Optional[str] = None,
        target_sequence: Optional[str] = None,
        target_language: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.source_sequence = source_sequence
        self.source_language = source_language
        self.target_sequence = target_sequence
        self.target_language = target_language

    def get_current_classification(self) -> Optional[str]:
        return self.target_sequence

    def update_classification(self, classification_result: str):
        self.target_sequence = classification_result

    def pretty_print(self, classification_result: Optional[str] = None) -> str:
        def maybe_prepend_language(text: str, lng: Optional[str]) -> str:
            return f"[{lng}] {text}" if lng is not None else text

        parts = [f"# input sequence: {maybe_prepend_language(self.source_sequence, self.source_language)}"]

        if self.target_sequence is not None:
            parts.append(f"\t# gold sequence: {maybe_prepend_language(self.target_sequence, self.target_language)}")

        if classification_result is not None:
            parts.append(
                f"\t# predicted sequence: {maybe_prepend_language(classification_result, self.target_language)}"
            )

        return "\n".join(parts)


class DataDriver:
    def read_from_path(
        self, path: str
    ) -> Generator[Union[SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample], None, None]:
        def r():
            with open(path) as f:
                for line in f:
                    yield line.strip()

        return self.read(r())

    def read(
        self, lines: Iterator[str]
    ) -> Generator[Union[SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample], None, None]:
        raise NotImplementedError

    def save(
        self,
        samples: Iterator[Union[SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample]],
        path: str,
    ):
        raise NotImplementedError


class SentencePairDataDriver(DataDriver):
    def read(self, lines: Iterator[str]) -> Generator[SentencePairSample, None, None]:
        raise NotImplementedError

    def save(self, samples: Iterator[SentencePairSample], path: str):
        raise NotImplementedError


class SequenceDataDriver(DataDriver):
    def read(self, lines: Iterator[str]) -> Generator[SequenceSample, None, None]:
        raise NotImplementedError

    def save(self, samples: Iterator[SequenceSample], path: str):
        raise NotImplementedError


class TokensDataDriver(DataDriver):
    def read(self, lines: Iterator[str]) -> Generator[TokensSample, None, None]:
        raise NotImplementedError

    def save(self, samples: Iterator[TokensSample], path: str):
        raise NotImplementedError


class QADataDriver(DataDriver):
    def read(self, lines: Iterator[str]) -> Generator[QASample, None, None]:
        raise NotImplementedError

    def save(self, samples: Iterator[QASample], path: str):
        raise NotImplementedError


class GenerationDataDriver(DataDriver):
    def read(self, lines: Iterator[str]) -> Generator[GenerationSample, None, None]:
        raise NotImplementedError

    def save(self, samples: Iterator[GenerationSample], path: str):
        raise NotImplementedError


class TSVSentencePairDataDriver(SentencePairDataDriver):
    def read(self, lines: Iterator[str]) -> Generator[SentencePairSample, None, None]:
        for line in lines:
            parts = line.split("\t")
            assert len(parts) in [
                2,
                3,
            ], f"TSVSentencePairDataDriver expects 2 (s1, s2) or 3 (s1, s2, label) fields, but {len(parts)} were found"
            sentence1, sentence2 = parts[0], parts[1]
            label = parts[2] if len(parts) == 3 else None
            yield SentencePairSample(sentence1, sentence2, label)

    def save(self, samples: Iterator[SentencePairSample], path: str):
        with open(path, "w") as f:
            for sample in samples:
                f.write(f"{sample.sentence1}\t{sample.sentence2}\t{sample.label}\n")


class JSONLSentencePairDataDriver(SentencePairDataDriver):
    def read(self, lines: Iterator[str]) -> Generator[SentencePairSample, None, None]:
        for line in lines:
            yield SentencePairSample(**json.loads(line))

    def save(self, samples: Iterator[SentencePairSample], path: str):
        with open(path, "w") as f:
            for sample in samples:
                f.write(
                    json.dumps(
                        {
                            "sentence1": sample.sentence1,
                            "sentence2": sample.sentence2,
                            **sample.get_additional_attributes(),
                            "label": sample.label,
                        }
                    )
                    + "\n"
                )


class TSVSequenceDataDriver(SequenceDataDriver):
    def read(self, lines: Iterator[str]) -> Generator[SequenceSample, None, None]:
        for line in lines:
            parts = line.split("\t")
            assert len(parts) in [
                1,
                2,
            ], f"TSVSequenceDataDriver expects 1 (sentence) or 2 (sentence, label) fields, but {len(parts)} were found at line {line}"
            sentence = parts[0]
            label = parts[1] if len(parts) == 2 else None
            yield SequenceSample(sentence, label)

    def save(self, samples: Iterator[SequenceSample], path: str):
        with open(path, "w") as f:
            for sample in samples:
                f.write(f"{sample.sequence}\t{sample.label}\n")


class JSONLSequenceDataDriver(SequenceDataDriver):
    def read(self, lines: Iterator[str]) -> Generator[SequenceSample, None, None]:
        for line in lines:
            yield SequenceSample(**json.loads(line))

    def save(self, samples: Iterator[SequenceSample], path: str):
        with open(path, "w") as f:
            for sample in samples:
                f.write(
                    json.dumps(
                        {"sequence": sample.sequence, **sample.get_additional_attributes(), "label": sample.label}
                    )
                    + "\n"
                )


class TSVTokensDataDriver(TokensDataDriver):
    def read(self, lines: Iterator[str]) -> Generator[TokensSample, None, None]:
        for line in lines:
            parts = line.split("\t")
            assert len(parts) in [
                1,
                2,
            ], f"TSVTokensDataDriver expects 1 (tokens) or 3 (tokens, labels) fields, but {len(parts)} were found at line {line}"
            tokens, labels = parts[0].split(" "), None
            if len(parts) == 2:
                labels = parts[1].split(" ")
                assert len(tokens) == len(
                    labels
                ), f"Token Classification requires as many token as labels: found {len(tokens)} tokens != {len(labels)} labels at line {line}"
            yield TokensSample(tokens, labels)

    def save(self, samples: Iterator[TokensSample], path: str):
        with open(path, "w") as f:
            for sample in samples:
                f.write("\t".join([" ".join(sample.tokens), " ".join(sample.labels)]) + "\n")


class JSONLTokensDataDriver(TokensDataDriver):
    def read(self, lines: Iterator[str]) -> Generator[TokensSample, None, None]:
        for line in lines:
            sample = TokensSample(**json.loads(line))
            if sample.labels is not None:
                if not sample.was_target_provided:
                    assert len(sample.tokens) == len(
                        sample.labels
                    ), f"Token Classification requires as many tokens as labels (please specify a target list if you need otherwise): found {len(sample.tokens)} tokens != {len(sample.labels)} labels at line {line}"
                else:
                    assert len(sample.target) == len(
                        sample.labels
                    ), f"Token Classification requires as many targets as labels: found {len(sample.target)} targets != {len(sample.labels)} labels at line {line}"
            yield TokensSample(**json.loads(line))

    def save(self, samples: Iterator[TokensSample], path: str):
        with open(path, "w") as f:
            for sample in samples:
                d = {
                    "tokens": sample.tokens,
                    "target": list(sample.target),
                    **sample.get_additional_attributes(),
                    "labels": sample.labels,
                }
                if sample.was_target_provided:
                    d["target"] = list(sample.target)
                f.write(json.dumps(d) + "\n")


class TSVQADataDriver(QADataDriver):
    def read(self, lines: Iterator[str]) -> Generator[QASample, None, None]:
        for i, line in enumerate(lines):
            parts = line.split("\t")
            assert len(parts) in [2, 4], (
                f"TSVQADataDriver expects 2 (context, question) or 4 (context, question, answer_start, answer_end) "
                f"fields, but {len(parts)} were found at line {i}: {line}"
            )
            context, question = parts[:2]
            answer_start, answer_end = None, None
            if len(parts) > 2:
                answer_start, answer_end = parts[2:]
                answer_start, answer_end = int(answer_start), int(answer_end)
            yield QASample(context, question, answer_start, answer_end)

    def save(self, samples: Iterator[QASample], path: str):
        with open(path, "w") as f:
            for sample in samples:
                sample_parts = [sample.context, sample.question]
                if sample.char_start is not None and sample.char_end is not None:
                    sample_parts += [str(sample.char_start), str(sample.char_end)]
                f.write("\t".join(sample_parts))
                f.write("\n")


class JSONLQADataDriver(QADataDriver):
    def read(self, lines: Iterator[str]) -> Generator[QASample, None, None]:
        for line in lines:
            yield QASample(**json.loads(line))

    def save(self, samples: Iterator[QASample], path: str):
        with open(path, "w") as f:
            for sample in samples:
                f.write(
                    json.dumps(
                        {
                            "question": sample.question,
                            "context": sample.context,
                            **sample.get_additional_attributes(),
                            "answer_start": sample.char_start,
                            "answer_end": sample.char_end,
                        }
                    )
                    + "\n"
                )


class TSVGenerationDataDriver(GenerationDataDriver):
    def read(self, lines: Iterator[str]) -> Generator[GenerationSample, None, None]:
        previous_line_parts = None
        for i, line in enumerate(lines):
            parts = line.split("\t")
            # sanity check
            assert len(parts) in [
                1,
                2,
            ], f"TSVGenerationDataDriver expects 1 (sequence) or 2 (input sequence, output sequence) fields, but {len(parts)} were found at line {i}: {line}"
            # consistency check
            if previous_line_parts is not None:
                assert len(parts) == len(
                    previous_line_parts
                ), f"TSVGenerationDataDriver expects all lines to contain the same number of fields, {len(parts)} fields at line {i} != {len(previous_line_parts)} fields at line {i-1}"
            previous_line_parts = parts
            yield GenerationSample(source_sequence=parts[0], target_sequence=parts[1] if len(parts) == 2 else None)

    def save(self, samples: Iterator[GenerationSample], path: str):
        with open(path, "w") as f:
            for sample in samples:
                assert (
                    sample.source_language is None and sample.target_language is None
                ), f"TSVGenerationDataDriver does not support language specification"
                f.write(f"{sample.source_sequence}\t{sample.target_sequence}\n")


class JSONLGenerationDataDriver(GenerationDataDriver):
    def read(self, lines: Iterator[str]) -> Generator[GenerationSample, None, None]:
        for line in lines:
            yield GenerationSample(**json.loads(line))

    def save(self, samples: Iterator[GenerationSample], path: str):
        with open(path, "w") as f:
            for sample in samples:
                f.write(
                    json.dumps(
                        {
                            "source_sequence": sample.source_sequence,
                            "source_language": sample.source_language,
                            "target_sequence": sample.target_sequence,
                            "target_language": sample.target_language,
                            **sample.get_additional_attributes(),
                        }
                    )
                    + "\n"
                )


# TASK TYPES
SEQUENCE = "sequence"
SENTENCE_PAIR = "sentence-pair"
TOKEN = "token"
QA = "qa"
GENERATION = "generation"

# FILE EXTENSIONS
TSV = "tsv"
JSONL = "jsonl"

READERS_DICT = {
    (SEQUENCE, TSV): TSVSequenceDataDriver,
    (SENTENCE_PAIR, TSV): TSVSentencePairDataDriver,
    (SEQUENCE, JSONL): JSONLSequenceDataDriver,
    (SENTENCE_PAIR, JSONL): JSONLSentencePairDataDriver,
    (TOKEN, JSONL): JSONLTokensDataDriver,
    (TOKEN, TSV): TSVTokensDataDriver,
    (QA, TSV): TSVQADataDriver,
    (QA, JSONL): JSONLQADataDriver,
    (GENERATION, TSV): TSVGenerationDataDriver,
    (GENERATION, JSONL): JSONLGenerationDataDriver,
}


@functools.lru_cache(maxsize=1_000)
def get_data_driver(task_type: str, file_extension: str) -> DataDriver:
    reader_identifier = (task_type, file_extension)
    if reader_identifier not in READERS_DICT:
        logger.info(f"No reader available for task {task_type} and extension {file_extension}.")
    assert reader_identifier in READERS_DICT, (
        f"Extension '{file_extension}' does not appear to be supported for task {task_type}. "
        f"Supported extensions are: {[e for t, e in READERS_DICT.keys() if t == task_type]}"
    )
    return READERS_DICT[reader_identifier]()
