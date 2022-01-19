from typing import List, Type, Union

from pydantic import BaseModel, Field

from classy.data.data_drivers import (
    GenerationSample,
    QASample,
    SentencePairSample,
    SequenceSample,
    TokensSample,
)


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


class MarshalInputSequenceSample(BaseModel, MarshalInputSample):
    sequence: str = Field(None, description="Input sequence")

    def unmarshal(self) -> SequenceSample:
        return SequenceSample(sequence=self.sequence)


class MarshalOutputSequenceSample(MarshalInputSequenceSample, MarshalOutputSample):
    label: str = Field(None, description="Label resulting from model classification")

    @classmethod
    def marshal(cls, sample: SequenceSample):
        return cls(sequence=sample.sequence, label=sample.predicted_annotation)


class SequenceTaskServeMixin(TaskServeMixin):
    @property
    def serve_input_class(self):
        return MarshalInputSequenceSample

    @property
    def serve_output_class(self):
        return MarshalOutputSequenceSample


class MarshalInputSentencePairSample(BaseModel, MarshalInputSample):
    sentence1: str = Field(None, description="First input sentence")
    sentence2: str = Field(None, description="Second input sentence")

    def unmarshal(self) -> SentencePairSample:
        return SentencePairSample(sentence1=self.sentence1, sentence2=self.sentence2)


class MarshalOutputSentencePairSample(
    MarshalInputSentencePairSample, MarshalOutputSample
):
    label: str = Field(None, description="Label resulting from model classification")

    @classmethod
    def marshal(cls, sample: SentencePairSample):
        return cls(
            sentence1=sample.sentence1,
            sentence2=sample.sentence2,
            label=sample.predicted_annotation,
        )


class SentencePairTaskServeMixin(TaskServeMixin):
    @property
    def serve_input_class(self):
        return MarshalInputSentencePairSample

    @property
    def serve_output_class(self):
        return MarshalOutputSentencePairSample


class MarshalInputTokensSample(BaseModel, MarshalInputSample):
    tokens: List[str] = Field(None, description="List of input tokens")

    def unmarshal(self) -> TokensSample:
        return TokensSample(tokens=self.tokens)


class MarshalOutputTokensSample(MarshalInputTokensSample, MarshalOutputSample):
    labels: List[str] = Field(
        None, description="List of labels the model assigned to each input token"
    )

    @classmethod
    def marshal(cls, sample: TokensSample):
        return cls(tokens=sample.tokens, labels=sample.predicted_annotation)


class TokenTaskServeMixin(TaskServeMixin):
    @property
    def serve_input_class(self):
        return MarshalInputTokensSample

    @property
    def serve_output_class(self):
        return MarshalOutputTokensSample


class MarshalInputQASample(BaseModel, MarshalInputSample):
    context: str = Field(None, description="Input context")
    question: str = Field(None, description="Input question")

    def unmarshal(self) -> QASample:
        return QASample(context=self.context, question=self.question)


class MarshalOutputQASample(MarshalInputQASample, MarshalOutputSample):
    answer_char_start: int = Field(None, description="Answer starting char index")
    answer_char_end: int = Field(None, description="Answer ending char index")

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
    def serve_input_class(self):
        return MarshalInputQASample

    @property
    def serve_output_class(self):
        return MarshalOutputQASample


class MarshalInputGenerationSample(BaseModel, MarshalInputSample):
    source_sequence: str = Field(None, description="Source sequence")
    source_language: str = Field(None, description="Source language")
    target_language: str = Field(None, description="Target language")

    def unmarshal(self) -> GenerationSample:
        return GenerationSample(
            source_sequence=self.source_sequence,
            source_language=self.source_language,
            target_language=self.target_language,
        )


class MarshalOutputGenerationSample(MarshalInputGenerationSample, MarshalOutputSample):
    target_sequence: str = Field(
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
    def serve_input_class(self):
        return MarshalInputGenerationSample

    @property
    def serve_output_class(self):
        return MarshalOutputGenerationSample
