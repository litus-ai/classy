import argparse
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, Body
from omegaconf import OmegaConf
from pydantic import BaseModel, Field

from classy.data.data_drivers import (
    SEQUENCE,
    TOKEN,
    SENTENCE_PAIR,
    QA,
    TokensSample,
    SentencePairSample,
    SequenceSample,
    QASample,
)
from classy.utils.commons import get_local_ip_address
from classy.utils.lightning import load_classy_module_from_checkpoint, load_prediction_dataset_conf_from_checkpoint
from classy.utils.log import get_project_logger

logger = get_project_logger(__name__)


class MarshalInputSequenceSample(BaseModel):
    sequence: str = Field(None, description="Input sequence")

    def unmarshal(self) -> SequenceSample:
        return SequenceSample(sequence=self.sequence)


class MarshalInputSentencePairSample(BaseModel):
    sentence1: str = Field(None, description="First input sentence")
    sentence2: str = Field(None, description="Second input sentence")

    def unmarshal(self) -> SentencePairSample:
        return SentencePairSample(sentence1=self.sentence1, sentence2=self.sentence2)


class MarshalInputTokensSample(BaseModel):
    tokens: List[str] = Field(None, description="List of input tokens")

    def unmarshal(self) -> TokensSample:
        return TokensSample(tokens=self.tokens)


class MarshalInputQASample(BaseModel):
    context: str = Field(None, description="Input context")
    question: str = Field(None, description="Input question")

    def unmarshal(self) -> QASample:
        return QASample(context=self.context, question=self.question)


class MarshalOutputSequenceSample(MarshalInputSequenceSample):
    label: str = Field(None, description="Label resulting from model classification")

    @classmethod
    def marshal(cls, sample: SequenceSample):
        return cls(sequence=sample.sequence, label=sample.label)


class MarshalOutputSentencePairSample(MarshalInputSentencePairSample):
    label: str = Field(None, description="Label resulting from model classification")

    @classmethod
    def marshal(cls, sample: SentencePairSample):
        return cls(sentence1=sample.sentence1, sentence2=sample.sentence2, label=sample.label)


class MarshalOutputTokensSample(MarshalInputTokensSample):
    labels: List[str] = Field(None, description="List of labels the model assigned to each input token")

    @classmethod
    def marshal(cls, sample: TokensSample):
        return cls(tokens=sample.tokens, labels=sample.labels)


class MarshalOutputQASample(MarshalInputQASample):
    answer_char_start: int = Field(None, description="Answer starting char index")
    answer_char_end: int = Field(None, description="Answer ending char index")

    @classmethod
    def marshal(cls, sample: QASample):
        return cls(
            context=sample.context,
            question=sample.question,
            answer_char_start=sample.char_start,
            answer_char_end=sample.char_end,
        )


def serve(
    model_checkpoint_path: str,
    port: int,
    cuda_device: int,
    token_batch_size: int,
    prediction_params: Optional[str] = None,
):

    # load model
    model = load_classy_module_from_checkpoint(model_checkpoint_path)
    model.to(torch.device(cuda_device if cuda_device != -1 else "cpu"))
    model.freeze()

    if prediction_params is not None:
        model.load_prediction_params(dict(OmegaConf.load(prediction_params)))

    # load dataset conf
    dataset_conf = load_prediction_dataset_conf_from_checkpoint(model_checkpoint_path)
    dataset_conf["_target_"] = dataset_conf["_target_"].replace(
        ".from_lines", ".from_samples"
    )  # todo can we do it better?

    # mock call to load resources
    next(model.predict(samples=[], dataset_conf=dataset_conf), None)

    # compute dynamic type
    if model.task == SEQUENCE:
        i_type, o_type = MarshalInputSequenceSample, MarshalOutputSequenceSample
    elif model.task == SENTENCE_PAIR:
        i_type, o_type = MarshalInputSentencePairSample, MarshalOutputSentencePairSample
    elif model.task == TOKEN:
        i_type, o_type = MarshalInputTokensSample, MarshalOutputTokensSample
    elif model.task == QA:
        i_type, o_type = MarshalInputQASample, MarshalOutputQASample
    else:
        raise ValueError()

    # for better readability on the OpenAPI docs
    # why leak the inner confusing class names
    class InputSample(i_type):
        pass

    class OutputSample(o_type):
        pass

    app = FastAPI(title="Classy Serve")

    @app.post("/", response_model=List[OutputSample], description="Prediction endpoint")
    def predict(input_samples: List[InputSample]) -> List[OutputSample]:

        output_samples = []

        for source, prediction in model.predict(
            model=model,
            samples=[input_sample.unmarshal() for input_sample in input_samples],
            dataset_conf=dataset_conf,
            token_batch_size=token_batch_size,
        ):
            source.update_classification(prediction)
            output_samples.append(OutputSample.marshal(source))

        return output_samples

    local_ip_address = get_local_ip_address()
    print(f"Model exposed at http://{local_ip_address}:{port}")
    print(f"Remember you can checkout the API at http://{local_ip_address}:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)


def main():
    args = parse_args()
    serve(
        model_checkpoint_path=args.model_checkpoint,
        prediction_params=args.prediction_params,
        port=args.p,
        cuda_device=args.cuda_device,
        token_batch_size=args.token_batch_size,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_checkpoint", type=str, help="Path to pl_modules checkpoint")
    parser.add_argument("--prediction-params", type=str, default=None, help="Path to prediction params")
    parser.add_argument("-p", type=int, default=8000, help="Port on which to expose the model")
    parser.add_argument("--cuda-device", type=int, default=-1, help="Cuda device")
    parser.add_argument("--token-batch-size", type=int, default=128, help="Token batch size")
    return parser.parse_args()


if __name__ == "__main__":
    main()
