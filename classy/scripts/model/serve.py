import argparse
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf

from classy.utils.commons import get_local_ip_address
from classy.utils.lightning import (
    load_classy_module_from_checkpoint,
    load_prediction_dataset_conf_from_checkpoint,
)
from classy.utils.log import get_project_logger

logger = get_project_logger(__name__)


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

    # mock call to load resources
    next(model.predict(samples=[], dataset_conf=dataset_conf), None)

    # for better readability on the OpenAPI docs
    # why leak the inner confusing class names
    class InputSample(model.serve_input_class):
        pass

    class OutputSample(model.serve_output_class):
        pass

    app = FastAPI(title="Classy Serve")

    @app.post("/", response_model=List[OutputSample], description="Prediction endpoint")
    def predict(input_samples: List[InputSample]) -> List[OutputSample]:

        output_samples = []

        for predicted_sample in model.predict(
            model=model,
            samples=[input_sample.unmarshal() for input_sample in input_samples],
            dataset_conf=dataset_conf,
            token_batch_size=token_batch_size,
        ):
            output_samples.append(OutputSample.marshal(predicted_sample))

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
    parser.add_argument(
        "model_checkpoint", type=str, help="Path to pl_modules checkpoint"
    )
    parser.add_argument(
        "--prediction-params", type=str, default=None, help="Path to prediction params"
    )
    parser.add_argument(
        "-p", type=int, default=8000, help="Port on which to expose the model"
    )
    parser.add_argument("--cuda-device", type=int, default=-1, help="Cuda device")
    parser.add_argument(
        "--token-batch-size", type=int, default=128, help="Token batch size"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
