import functools
import json
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

from classy.utils.log import get_project_logger

logger = get_project_logger(__name__)


class ClassySample:
    reference_annotation: Optional
    predicted_annotation: Optional

    def __init__(self, **kwargs):
        super().__setattr__("_d", {})
        self._d = kwargs

    def _get_reference_annotation(
        self,
    ) -> Optional[Union[str, List[str], Tuple[int, int]]]:
        raise NotImplementedError

    def _update_reference_annotation(
        self, reference_annotation: Union[str, List[str], Tuple[int, int]]
    ):
        raise NotImplementedError

    def _get_predicted_annotation(
        self,
    ) -> Optional[Union[str, List[str], Tuple[int, int]]]:
        raise NotImplementedError

    def _update_predicted_annotation(
        self, predicted_annotation: Union[str, List[str], Tuple[int, int]]
    ):
        raise NotImplementedError

    def __getattribute__(self, item):
        if item == "reference_annotation":
            return self._get_reference_annotation()
        elif item == "predicted_annotation":
            return self._get_predicted_annotation()
        else:
            return super(ClassySample, self).__getattribute__(item)

    def __getattr__(self, item):
        if item.startswith("__") and item.startswith("__"):
            # this is likely some python library-specific variable (such as __deepcopy__ for copy)
            # better follow standard behavior here
            raise AttributeError(item)
        elif item in self._d:
            return self._d[item]
        else:
            return None

    def __setattr__(self, key, value):
        if key == "reference_annotation":
            return self._update_reference_annotation(value)
        elif key == "predicted_annotation":
            return self._update_predicted_annotation(value)
        if key in self._d:
            self._d[key] = value
        else:
            super().__setattr__(key, value)

    def get_additional_attributes(self) -> Dict[Any, Any]:
        return self._d

    def pretty_print(self) -> str:
        raise NotImplementedError

    @property
    def input(self) -> str:
        raise NotImplementedError


class SentencePairSample(ClassySample):
    def __init__(
        self, sentence1: str, sentence2: str, label: Optional[str] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self._reference_annotation = label
        self._predicted_annotation = None

    def _get_reference_annotation(self) -> Optional[str]:
        return self._reference_annotation

    def _update_reference_annotation(self, reference_annotation: str):
        self._reference_annotation = reference_annotation

    def _get_predicted_annotation(self) -> Optional[str]:
        return self._predicted_annotation

    def _update_predicted_annotation(self, predicted_annotation: str):
        self._predicted_annotation = predicted_annotation

    def pretty_print(self) -> str:
        parts = [f"# sentence1: {self.sentence1}", f"# sentence2: {self.sentence2}"]
        if self.reference_annotation is not None:
            parts.append(f"\t# label: {self.reference_annotation}")
        if self.predicted_annotation is not None:
            parts.append(f"\t# classification_result: {self.predicted_annotation}")
        return "\n".join(parts)

    @property
    def input(self) -> str:
        return f"{self.sentence1} ==> {self.sentence2}"


class SequenceSample(ClassySample):
    def __init__(self, sequence: str, label: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.sequence = sequence
        self._reference_annotation = label
        self._predicted_annotation = None

    def _get_reference_annotation(self) -> Optional[str]:
        return self._reference_annotation

    def _update_reference_annotation(self, reference_annotation: str):
        self._reference_annotation = reference_annotation

    def _get_predicted_annotation(self) -> Optional[str]:
        return self._predicted_annotation

    def _update_predicted_annotation(self, predicted_annotation: str):
        self._predicted_annotation = predicted_annotation

    def pretty_print(self) -> str:
        parts = [f"# sequence: {self.sequence}"]
        if self.reference_annotation is not None:
            parts.append(f"\t# label: {self.reference_annotation}")
        if self.predicted_annotation is not None:
            parts.append(f"\t# classification_result: {self.predicted_annotation}")
        return "\n".join(parts)

    @property
    def input(self) -> str:
        return self.sequence


class TokensSample(ClassySample):
    def __init__(
        self,
        tokens: List[str],
        labels: Optional[List[str]] = None,
        target: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokens = tokens
        self._reference_annotation = labels
        self._predicted_annotation = None
        self.target = target if target is not None else list(range(len(tokens)))
        self.was_target_provided = target is not None

    def _get_reference_annotation(self) -> Optional[List[str]]:
        return self._reference_annotation

    def _update_reference_annotation(self, reference_annotation: List[str]):
        self._reference_annotation = reference_annotation

    def _get_predicted_annotation(self) -> Optional[List[str]]:
        return self._predicted_annotation

    def _update_predicted_annotation(self, predicted_annotation: List[str]):
        self._predicted_annotation = predicted_annotation

    def pretty_print(self) -> str:
        parts = [f'# tokens: {" ".join(self.tokens)}']
        if self.reference_annotation is not None:
            parts.append(f'\t# labels: {" ".join(self.reference_annotation)}')
        if self.predicted_annotation is not None:
            parts.append(
                f'\t# classification_result: {" ".join(self.predicted_annotation)}'
            )
        return "\n".join(parts)

    @property
    def input(self) -> str:
        return " ".join(self.tokens)


class QASample(ClassySample):
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
        if answer_start is not None and answer_end is not None:
            self._reference_annotation = answer_start, answer_end
        else:
            self._reference_annotation = None
        self._predicted_annotation = None

    def _get_reference_annotation(self) -> Optional[Tuple[int, int]]:
        return self._reference_annotation

    def _update_reference_annotation(self, reference_annotation: Tuple[int, int]):
        self._reference_annotation = reference_annotation

    def _get_predicted_annotation(self) -> Optional[Tuple[int, int]]:
        return self._predicted_annotation

    def _update_predicted_annotation(self, predicted_annotation: Tuple[int, int]):
        self._predicted_annotation = predicted_annotation

    def pretty_print(self) -> str:
        parts = [
            f"# context: {self.context}",
            f"# question: {self.question}",
        ]

        if self.reference_annotation is not None:
            char_start, char_end = self.reference_annotation
            parts += [
                "### Gold positions ###",
                f"# answer start: {char_start}, answer end: {char_end}",
                f"# answer: {self.context[char_start:char_end]}",
            ]

        if self.predicted_annotation is not None:
            classification_start, classification_end = self.predicted_annotation
            parts += [
                "### Predicted positions ###",
                f"# answer start: {classification_start}, answer end: {classification_end}",
                f"# answer: {self.context[classification_start:classification_end]}",
            ]

        return "\n".join(parts)

    @property
    def input(self) -> str:
        return f"[Q] {self.question} [C] {self.context}"


class GenerationSample(ClassySample):
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
        self.target_language = target_language
        self._reference_annotation = target_sequence
        self._predicted_annotation = None

    def _get_reference_annotation(self) -> Optional[str]:
        return self._reference_annotation

    def _update_reference_annotation(self, reference_annotation: str):
        self._reference_annotation = reference_annotation

    def _get_predicted_annotation(self) -> Optional[str]:
        return self._predicted_annotation

    def _update_predicted_annotation(self, predicted_annotation: str):
        self._predicted_annotation = predicted_annotation

    def pretty_print(self) -> str:
        def maybe_prepend_language(text: str, lng: Optional[str]) -> str:
            return f"[{lng}] {text}" if lng is not None else text

        parts = [
            f"# input sequence: {maybe_prepend_language(self.source_sequence, self.source_language)}"
        ]

        if self.reference_annotation is not None:
            parts.append(
                f"\t# gold sequence: {maybe_prepend_language(self.reference_annotation, self.target_language)}"
            )

        if self.predicted_annotation is not None:
            parts.append(
                f"\t# predicted sequence: {maybe_prepend_language(self.predicted_annotation, self.target_language)}"
            )

        return "\n".join(parts)

    @property
    def input(self) -> str:
        if self.source_language is None:
            return self.source_sequence
        else:
            return f"[{self.source_language} => {self.target_language}] {self.source_sequence}"


class DataDriver:
    def dataset_exists_at_path(self, path: str) -> bool:
        return Path(path).exists()

    def read_from_path(self, path: str) -> Iterator[ClassySample]:
        def r():
            with open(path) as f:
                for line in f:
                    yield line.strip()

        return self.read(r())

    def read(self, lines: Iterator[str]) -> Iterator[ClassySample]:
        raise NotImplementedError

    def save(
        self,
        samples: Iterator[ClassySample],
        path: str,
        use_predicted_annotation: bool = False,
    ):
        raise NotImplementedError


class SentencePairDataDriver(DataDriver):
    def read(self, lines: Iterator[str]) -> Iterator[SentencePairSample]:
        raise NotImplementedError

    def save(
        self,
        samples: Iterator[SentencePairSample],
        path: str,
        use_predicted_annotation: bool = False,
    ):
        raise NotImplementedError


class SequenceDataDriver(DataDriver):
    def read(self, lines: Iterator[str]) -> Iterator[SequenceSample]:
        raise NotImplementedError

    def save(
        self,
        samples: Iterator[SequenceSample],
        path: str,
        use_predicted_annotation: bool = False,
    ):
        raise NotImplementedError


class TokensDataDriver(DataDriver):
    def read(self, lines: Iterator[str]) -> Iterator[TokensSample]:
        raise NotImplementedError

    def save(
        self,
        samples: Iterator[TokensSample],
        path: str,
        use_predicted_annotation: bool = False,
    ):
        raise NotImplementedError


class QADataDriver(DataDriver):
    def read(self, lines: Iterator[str]) -> Iterator[QASample]:
        raise NotImplementedError

    def save(
        self,
        samples: Iterator[QASample],
        path: str,
        use_predicted_annotation: bool = False,
    ):
        raise NotImplementedError


class GenerationDataDriver(DataDriver):
    def read(self, lines: Iterator[str]) -> Iterator[GenerationSample]:
        raise NotImplementedError

    def save(
        self,
        samples: Iterator[GenerationSample],
        path: str,
        use_predicted_annotation: bool = False,
    ):
        raise NotImplementedError


class TSVSentencePairDataDriver(SentencePairDataDriver):
    def read(self, lines: Iterator[str]) -> Iterator[SentencePairSample]:
        for line in lines:
            parts = line.split("\t")
            assert len(parts) in [
                2,
                3,
            ], f"TSVSentencePairDataDriver expects 2 (s1, s2) or 3 (s1, s2, label) fields, but {len(parts)} were found"
            sentence1, sentence2 = parts[0], parts[1]
            label = parts[2] if len(parts) == 3 else None
            yield SentencePairSample(sentence1, sentence2, label)

    def save(
        self,
        samples: Iterator[SentencePairSample],
        path: str,
        use_predicted_annotation: bool = False,
    ):
        with open(path, "w") as f:
            for sample in samples:
                used_annotation = (
                    sample.predicted_annotation
                    if use_predicted_annotation
                    else sample.reference_annotation
                )
                f.write(f"{sample.sentence1}\t{sample.sentence2}\t{used_annotation}\n")


class JSONLSentencePairDataDriver(SentencePairDataDriver):
    def read(self, lines: Iterator[str]) -> Iterator[SentencePairSample]:
        for line in lines:
            yield SentencePairSample(**json.loads(line))

    def save(
        self,
        samples: Iterator[SentencePairSample],
        path: str,
        use_predicted_annotation: bool = False,
    ):
        with open(path, "w") as f:
            for sample in samples:
                used_annotation = (
                    sample.predicted_annotation
                    if use_predicted_annotation
                    else sample.reference_annotation
                )
                f.write(
                    json.dumps(
                        {
                            "sentence1": sample.sentence1,
                            "sentence2": sample.sentence2,
                            **sample.get_additional_attributes(),
                            "label": used_annotation,
                        }
                    )
                    + "\n"
                )


class TSVSequenceDataDriver(SequenceDataDriver):
    def read(self, lines: Iterator[str]) -> Iterator[SequenceSample]:
        for line in lines:
            parts = line.split("\t")
            assert len(parts) in [
                1,
                2,
            ], f"TSVSequenceDataDriver expects 1 (sentence) or 2 (sentence, label) fields, but {len(parts)} were found at line {line}"
            sentence = parts[0]
            label = parts[1] if len(parts) == 2 else None
            yield SequenceSample(sentence, label)

    def save(
        self,
        samples: Iterator[SequenceSample],
        path: str,
        use_predicted_annotation: bool = False,
    ):
        with open(path, "w") as f:
            for sample in samples:
                used_annotation = (
                    sample.predicted_annotation
                    if use_predicted_annotation
                    else sample.reference_annotation
                )
                f.write(f"{sample.sequence}\t{used_annotation}\n")


class JSONLSequenceDataDriver(SequenceDataDriver):
    def read(self, lines: Iterator[str]) -> Iterator[SequenceSample]:
        for line in lines:
            yield SequenceSample(**json.loads(line))

    def save(
        self,
        samples: Iterator[SequenceSample],
        path: str,
        use_predicted_annotation: bool = False,
    ):
        with open(path, "w") as f:
            for sample in samples:
                used_annotation = (
                    sample.predicted_annotation
                    if use_predicted_annotation
                    else sample.reference_annotation
                )
                f.write(
                    json.dumps(
                        {
                            "sequence": sample.sequence,
                            **sample.get_additional_attributes(),
                            "label": used_annotation,
                        }
                    )
                    + "\n"
                )


class TSVTokensDataDriver(TokensDataDriver):
    def read(self, lines: Iterator[str]) -> Iterator[TokensSample]:
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

    def save(
        self,
        samples: Iterator[TokensSample],
        path: str,
        use_predicted_annotation: bool = False,
    ):
        with open(path, "w") as f:
            for sample in samples:
                used_annotation = (
                    sample.predicted_annotation
                    if use_predicted_annotation
                    else sample.reference_annotation
                )
                f.write(
                    "\t".join([" ".join(sample.tokens), " ".join(used_annotation)])
                    + "\n"
                )


class JSONLTokensDataDriver(TokensDataDriver):
    def read(self, lines: Iterator[str]) -> Iterator[TokensSample]:
        for line in lines:
            sample = TokensSample(**json.loads(line))
            if sample.reference_annotation is not None:
                if not sample.was_target_provided:
                    assert len(sample.tokens) == len(
                        sample.reference_annotation
                    ), f"Token Classification requires as many tokens as labels (please specify a target list if you need otherwise): found {len(sample.tokens)} tokens != {len(sample.reference_annotation)} labels at line {line}"
                else:
                    assert len(sample.target) == len(
                        sample.reference_annotation
                    ), f"Token Classification requires as many targets as labels: found {len(sample.target)} targets != {len(sample.reference_annotation)} labels at line {line}"
            yield sample

    def save(
        self,
        samples: Iterator[TokensSample],
        path: str,
        use_predicted_annotation: bool = False,
    ):
        with open(path, "w") as f:
            for sample in samples:
                used_annotation = (
                    sample.predicted_annotation
                    if use_predicted_annotation
                    else sample.reference_annotation
                )
                d = {
                    "tokens": sample.tokens,
                    **sample.get_additional_attributes(),
                    "labels": used_annotation,
                }
                if sample.was_target_provided:
                    d["target"] = list(sample.target)
                f.write(json.dumps(d) + "\n")


class TSVQADataDriver(QADataDriver):
    def read(self, lines: Iterator[str]) -> Iterator[QASample]:
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

    def save(
        self,
        samples: Iterator[QASample],
        path: str,
        use_predicted_annotation: bool = False,
    ):
        with open(path, "w") as f:
            for sample in samples:
                sample_parts = [sample.context, sample.question]
                used_annotation = (
                    sample.predicted_annotation
                    if use_predicted_annotation
                    else sample.reference_annotation
                )
                if used_annotation is not None:
                    char_start, char_end = used_annotation
                    sample_parts += [str(char_start), str(char_end)]
                f.write("\t".join(sample_parts))
                f.write("\n")


class JSONLQADataDriver(QADataDriver):
    def read(self, lines: Iterator[str]) -> Iterator[QASample]:
        for line in lines:
            yield QASample(**json.loads(line))

    def save(
        self,
        samples: Iterator[QASample],
        path: str,
        use_predicted_annotation: bool = False,
    ):
        with open(path, "w") as f:
            for sample in samples:
                char_start, char_end = None, None
                used_annotation = (
                    sample.predicted_annotation
                    if use_predicted_annotation
                    else sample.reference_annotation
                )
                if used_annotation is not None:
                    char_start, char_end = used_annotation
                f.write(
                    json.dumps(
                        {
                            "question": sample.question,
                            "context": sample.context,
                            **sample.get_additional_attributes(),
                            "answer_start": char_start,
                            "answer_end": char_end,
                        }
                    )
                    + "\n"
                )


class TSVGenerationDataDriver(GenerationDataDriver):
    def read(self, lines: Iterator[str]) -> Iterator[GenerationSample]:
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
            yield GenerationSample(
                source_sequence=parts[0],
                target_sequence=parts[1] if len(parts) == 2 else None,
            )

    def save(
        self,
        samples: Iterator[GenerationSample],
        path: str,
        use_predicted_annotation: bool = False,
    ):
        with open(path, "w") as f:
            for sample in samples:
                assert (
                    sample.source_language is None and sample.target_language is None
                ), f"TSVGenerationDataDriver does not support language specification"
                used_annotation = (
                    sample.predicted_annotation
                    if use_predicted_annotation
                    else sample.reference_annotation
                )
                if used_annotation is None:
                    f.write(f"{sample.source_sequence}\n")
                else:
                    f.write(f"{sample.source_sequence}\t{used_annotation}\n")


class JSONLGenerationDataDriver(GenerationDataDriver):
    def read(self, lines: Iterator[str]) -> Iterator[GenerationSample]:
        for line in lines:
            yield GenerationSample(**json.loads(line))

    def save(
        self,
        samples: Iterator[GenerationSample],
        path: str,
        use_predicted_annotation: bool = False,
    ):
        with open(path, "w") as f:
            for sample in samples:
                used_annotation = (
                    sample.predicted_annotation
                    if use_predicted_annotation
                    else sample.reference_annotation
                )
                f.write(
                    json.dumps(
                        {
                            "source_sequence": sample.source_sequence,
                            "source_language": sample.source_language,
                            "target_sequence": used_annotation,
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
        logger.info(
            f"No reader available for task {task_type} and extension {file_extension}."
        )
    assert reader_identifier in READERS_DICT, (
        f"Extension '{file_extension}' does not appear to be supported for task {task_type}. "
        f"Supported extensions are: {[e for t, e in READERS_DICT.keys() if t == task_type]}"
    )
    return READERS_DICT[reader_identifier]()
