import argparse
import hydra
from typing import Optional, List, Callable, Union, Tuple, Dict

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.multiclass import unique_labels

from classy.data.data_drivers import get_data_driver, SEQUENCE, SENTENCE_PAIR, TOKEN, QA
from classy.data.data_drivers import SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample
from classy.utils.commons import flatten
from classy.utils.lightning import load_classy_module_from_checkpoint, load_prediction_dataset_conf_from_checkpoint, \
    load_training_conf_from_checkpoint


def evaluate(
    model_checkpoint_path: str,
    cuda_device: int,
    token_batch_size: int,
    input_path: str,
    output_path: Optional[str] = None,
    prediction_params: Optional[str] = None,
    metrics_fn: Optional[Callable[[List[Tuple[Union[SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample], Union[str, List[str]]]]], Dict]] = None,
):

    # load model
    model = load_classy_module_from_checkpoint(model_checkpoint_path)
    model.to(torch.device(cuda_device if cuda_device != -1 else "cpu"))
    model.freeze()

    if prediction_params is not None:
        model.load_prediction_params(dict(OmegaConf.load(prediction_params)))

    # load dataset conf and driver
    dataset_conf = load_prediction_dataset_conf_from_checkpoint(model_checkpoint_path)
    input_extension = input_path.split(".")[-1]
    data_driver = get_data_driver(model.task, input_extension)

    # load metrics_fn if None
    if metrics_fn is None:
        evaluation_conf = load_training_conf_from_checkpoint(model_checkpoint_path).evaluation
        metrics_fn = hydra.utils.instantiate(evaluation_conf)

    # predict
    predicted_samples = list(
        model.predict(
            model=model,
            samples=data_driver.read_from_path(input_path),
            dataset_conf=dataset_conf,
            token_batch_size=token_batch_size,
            progress_bar=True,
        )
    )

    # dump predictions if requested
    if output_path is not None:
        with open(output_path, "w") as f:
            for sample, p in predicted_samples:
                f.write(sample.pretty_print(classification_result=p) + "\n")

    # run evaluation and print metrics
    result = metrics_fn(predicted_samples)
    for metric_name, metric_f in result.items():
        print(f'* {metric_name}: {metric_f}')


def main():
    args = parse_args()
    evaluate(
        model_checkpoint_path=args.model_checkpoint,
        cuda_device=args.cuda_device,
        token_batch_size=args.token_batch_size,
        input_path=args.f,
        output_path=args.o,
        prediction_params=args.prediction_params,
        metrics_fn=None,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    # prediction args
    parser.add_argument("model_checkpoint", type=str, help="Path to pl_modules checkpoint")
    parser.add_argument("--prediction-params", type=str, default=None, help="Path to prediction params")
    parser.add_argument("--cuda-device", type=int, default=-1, help="Cuda device")
    parser.add_argument("--token-batch-size", type=int, default=128, help="Token batch size")
    # evaluation args
    parser.add_argument("-f", type=str, required=True, help="Dataset to evaluate upon")
    parser.add_argument("-o", type=str, default=None, help="If given, predictions will be dumped at the provided file")
    # return
    return parser.parse_args()


if __name__ == "__main__":
    main()
