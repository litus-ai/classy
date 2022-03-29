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


class MarshalInputSequenceSample(pydantic.BaseModel, MarshalInputSample):
    sequence: str = pydantic.Field(None, description="Input sequence")

    def unmarshal(self) -> SequenceSample:
        return SequenceSample(sequence=self.sequence)


class MarshalOutputSequenceSample(MarshalInputSequenceSample, MarshalOutputSample):
    label: str = pydantic.Field(
        None, description="Label resulting from model classification"
    )

    @classmethod
    def marshal(cls, sample: SequenceSample):
        return cls(sequence=sample.sequence, label=sample.predicted_annotation)


class SequenceTaskServeMixin(TaskServeMixin):
    @property
    @requires("streamlit")
    def serve_input_class(self):
        return MarshalInputSequenceSample

    @property
    @requires("streamlit")
    def serve_output_class(self):
        return MarshalOutputSequenceSample


class MarshalInputSentencePairSample(pydantic.BaseModel, MarshalInputSample):
    sentence1: str = pydantic.Field(None, description="First input sentence")
    sentence2: str = pydantic.Field(None, description="Second input sentence")

    def unmarshal(self) -> SentencePairSample:
        return SentencePairSample(sentence1=self.sentence1, sentence2=self.sentence2)


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


class SentencePairTaskServeMixin(TaskServeMixin):
    @property
    @requires("streamlit")
    def serve_input_class(self):
        return MarshalInputSentencePairSample

    @property
    @requires("streamlit")
    def serve_output_class(self):
        return MarshalOutputSentencePairSample


class MarshalInputTokensSample(pydantic.BaseModel, MarshalInputSample):
    tokens: List[str] = pydantic.Field(None, description="List of input tokens")

    def unmarshal(self) -> TokensSample:
        return TokensSample(tokens=self.tokens)


class MarshalOutputTokensSample(MarshalInputTokensSample, MarshalOutputSample):
    labels: List[str] = pydantic.Field(
        None, description="List of labels the model assigned to each input token"
    )

    @classmethod
    def marshal(cls, sample: TokensSample):
        return cls(tokens=sample.tokens, labels=sample.predicted_annotation)


class TokenTaskServeMixin(TaskServeMixin):
    @property
    @requires("streamlit")
    def serve_input_class(self):
        return MarshalInputTokensSample

    @property
    @requires("streamlit")
    def serve_output_class(self):
        return MarshalOutputTokensSample


class MarshalInputQASample(pydantic.BaseModel, MarshalInputSample):
    context: str = pydantic.Field(None, description="Input context")
    question: str = pydantic.Field(None, description="Input question")

    def unmarshal(self) -> QASample:
        return QASample(context=self.context, question=self.question)


class MarshalOutputQASample(MarshalInputQASample, MarshalOutputSample):
    answer_char_start: int = pydantic.Field(
        None, description="Answer starting char index"
    )
    answer_char_end: int = pydantic.Field(None, description="Answer ending char index")

    @classmethod
    def marshal(cls, sample: QASample):
        char_start, char_end = sample.predicted_annotation
        return cls(
            context=sample.context,
            question=sample.question,
            answer_char_start=char_start,
            answer_char_end=char_end,
        )


class QATaskServeMixin(TaskServeMixin):
    @property
    @requires("streamlit")
    def serve_input_class(self):
        return MarshalInputQASample

    @property
    @requires("streamlit")
    def serve_output_class(self):
        return MarshalOutputQASample


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


class MarshalOutputGenerationSample(MarshalInputGenerationSample, MarshalOutputSample):
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


class GenerationTaskServeMixin(TaskServeMixin):
    @property
    @requires("streamlit")
    def serve_input_class(self):
        return MarshalInputGenerationSample

    @property
    @requires("streamlit")
    def serve_output_class(self):
        return MarshalOutputGenerationSample
