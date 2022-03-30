from typing import List, Type, Union

from classy.data.data_drivers import (
    GenerationSample,
    QASample,
    SentencePairSample,
    SequenceSample,
    TokensSample,
)
from classy.utils.optional_deps import requires

try:
    import pydantic
except ImportError:
    pydantic = None


class MarshalInputSample:
    def unmarshal(
        self,
    ) -> Union[
        SequenceSample, SentencePairSample, TokensSample, QASample, GenerationSample
    ]:
        raise NotImplementedError


class MarshalOutputSample:
    @classmethod
    def marshal(
        cls,
        sample: Union[
            SequenceSample, SentencePairSample, TokensSample, QASample, GenerationSample
        ],
    ):
        raise NotImplementedError


class TaskServeMixin:
    @property
    def serve_input_class(self) -> Type[MarshalInputSample]:
        raise NotImplementedError

    @property
    def serve_output_class(self) -> Type[MarshalOutputSample]:
        raise NotImplementedError


class SequenceTaskServeMixin(TaskServeMixin):

    if pydantic is not None:

        class MarshalInputSequenceSample(pydantic.BaseModel, MarshalInputSample):
            sequence: str = pydantic.Field(None, description="Input sequence")

            def unmarshal(self) -> SequenceSample:
                return SequenceSample(sequence=self.sequence)

        class MarshalOutputSequenceSample(
            MarshalInputSequenceSample, MarshalOutputSample
        ):
            label: str = pydantic.Field(
                None, description="Label resulting from model classification"
            )

            @classmethod
            def marshal(cls, sample: SequenceSample):
                return cls(sequence=sample.sequence, label=sample.predicted_annotation)

    else:

        MarshalInputSequenceSample, MarshalOutputSequenceSample = None, None

    @property
    @requires("pydantic")
    def serve_input_class(self):
        return SequenceTaskServeMixin.MarshalInputSequenceSample

    @property
    @requires("pydantic")
    def serve_output_class(self):
        return SequenceTaskServeMixin.MarshalOutputSequenceSample


class SentencePairTaskServeMixin(TaskServeMixin):

    if pydantic is not None:

        class MarshalInputSentencePairSample(pydantic.BaseModel, MarshalInputSample):
            sentence1: str = pydantic.Field(None, description="First input sentence")
            sentence2: str = pydantic.Field(None, description="Second input sentence")

            def unmarshal(self) -> SentencePairSample:
                return SentencePairSample(
                    sentence1=self.sentence1, sentence2=self.sentence2
                )

        class MarshalOutputSentencePairSample(
            MarshalInputSentencePairSample, MarshalOutputSample
        ):
            label: str = pydantic.Field(
                None, description="Label resulting from model classification"
            )

            @classmethod
            def marshal(cls, sample: SentencePairSample):
                return cls(
                    sentence1=sample.sentence1,
                    sentence2=sample.sentence2,
                    label=sample.predicted_annotation,
                )

    else:

        MarshalInputSentencePairSample, MarshalOutputSentencePairSample = None, None

    @property
    @requires("pydantic")
    def serve_input_class(self):
        return SentencePairTaskServeMixin.MarshalInputSentencePairSample

    @property
    @requires("pydantic")
    def serve_output_class(self):
        return SentencePairTaskServeMixin.MarshalOutputSentencePairSample


class TokenTaskServeMixin(TaskServeMixin):

    if pydantic is not None:

        class MarshalInputTokensSample(pydantic.BaseModel, MarshalInputSample):
            tokens: List[str] = pydantic.Field(None, description="List of input tokens")

            def unmarshal(self) -> TokensSample:
                return TokensSample(tokens=self.tokens)

        class MarshalOutputTokensSample(MarshalInputTokensSample, MarshalOutputSample):
            labels: List[str] = pydantic.Field(
                None,
                description="List of labels the model assigned to each input token",
            )

            @classmethod
            def marshal(cls, sample: TokensSample):
                return cls(tokens=sample.tokens, labels=sample.predicted_annotation)

    else:

        MarshalInputTokensSample, MarshalOutputTokensSample = None, None

    @property
    @requires("pydantic")
    def serve_input_class(self):
        return TokenTaskServeMixin.MarshalInputTokensSample

    @property
    @requires("pydantic")
    def serve_output_class(self):
        return TokenTaskServeMixin.MarshalOutputTokensSample


class QATaskServeMixin(TaskServeMixin):

    if pydantic is not None:

        class MarshalInputQASample(pydantic.BaseModel, MarshalInputSample):
            context: str = pydantic.Field(None, description="Input context")
            question: str = pydantic.Field(None, description="Input question")

            def unmarshal(self) -> QASample:
                return QASample(context=self.context, question=self.question)

        class MarshalOutputQASample(MarshalInputQASample, MarshalOutputSample):
            answer_char_start: int = pydantic.Field(
                None, description="Answer starting char index"
            )
            answer_char_end: int = pydantic.Field(
                None, description="Answer ending char index"
            )

            @classmethod
            def marshal(cls, sample: QASample):
                char_start, char_end = sample.predicted_annotation
                return cls(
                    context=sample.context,
                    question=sample.question,
                    answer_char_start=char_start,
                    answer_char_end=char_end,
                )

    else:

        MarshalInputQASample, MarshalOutputQASample = None, None

    @property
    @requires("pydantic")
    def serve_input_class(self):
        return QATaskServeMixin.MarshalInputQASample

    @property
    @requires("pydantic")
    def serve_output_class(self):
        return QATaskServeMixin.MarshalOutputQASample


class GenerationTaskServeMixin(TaskServeMixin):

    if pydantic is not None:

        class MarshalInputGenerationSample(pydantic.BaseModel, MarshalInputSample):
            source_sequence: str = pydantic.Field(None, description="Source sequence")
            source_language: str = pydantic.Field(None, description="Source language")
            target_language: str = pydantic.Field(None, description="Target language")

            def unmarshal(self) -> GenerationSample:
                return GenerationSample(
                    source_sequence=self.source_sequence,
                    source_language=self.source_language,
                    target_language=self.target_language,
                )

        class MarshalOutputGenerationSample(
            MarshalInputGenerationSample, MarshalOutputSample
        ):
            target_sequence: str = pydantic.Field(
                None, description="Target sequence resulting from model classification"
            )

            @classmethod
            def marshal(cls, sample: GenerationSample):
                return cls(
                    source_sequence=sample.source_sequence,
                    source_language=sample.source_language,
                    target_language=sample.target_language,
                    target_sequence=sample.predicted_annotation,
                )

    else:

        MarshalInputGenerationSample, MarshalOutputGenerationSample = None, None

    @property
    @requires("pydantic")
    def serve_input_class(self):
        return GenerationTaskServeMixin.MarshalInputGenerationSample

    @property
    @requires("pydantic")
    def serve_output_class(self):
        return GenerationTaskServeMixin.MarshalOutputGenerationSample
