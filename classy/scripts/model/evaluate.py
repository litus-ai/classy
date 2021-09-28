import argparse
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
from classy.utils.commons import flatten
from classy.utils.lightning import load_classy_module_from_checkpoint, load_prediction_dataset_conf_from_checkpoint


def get_default_metrics(
    task: str,
) -> Dict[str, Callable[[List[Union[Tuple[str, str], Tuple[List[str], List[str]]]]], None]]:
    def preprocess(labels, task):
        if task == QA:
            gold, pred = [str(l[0]) for l in labels], [str(l[1]) for l in labels]
            return gold, pred

        gold, pred = [l[0] for l in labels], [l[1] for l in labels]
        if task == TOKEN:
            gold, pred = flatten(gold), flatten(pred)
        return gold, pred

    def accuracy(gold, pred):
        print(f"# accuracy: {accuracy_score(gold, pred):.4f}")

    def f1(gold, pred):
        print(f"# f1-score: {f1_score(gold, pred, average='micro')}")

    def p_r_f_support(gold, pred):
        parts = [f"# classification metrics"]
        for avg in ["micro", "macro", "weighted"]:
            parts.append(f"\t# avg strategy: {avg}")
            p, r, f1, _ = precision_recall_fscore_support(gold, pred, average=avg)
            for k, v in zip(["precision", "recall", "f1"], [p, r, f1]):
                parts.append(f"\t\t# {k}: {v:.4f}")
        print("\n".join(parts))

    def plot_confusion_matrix(gold, pred):
        cm = confusion_matrix(gold, pred, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels(gold, pred))
        disp.plot(cmap=plt.cm.Blues, values_format=".2f")
        plt.show()

    if task == QA:
        return {
            "f1": lambda labels: f1(*preprocess(labels, task=task)),
        }
    return {
        "accuracy": lambda labels: accuracy(*preprocess(labels, task=task)),
        "classification metrics": lambda labels: p_r_f_support(*preprocess(labels, task=task)),
        "confusion matrix": lambda labels: plot_confusion_matrix(*preprocess(labels, task=task)),
    }


def evaluate(
    model_checkpoint_path: str,
    cuda_device: int,
    token_batch_size: int,
    input_path: str,
    output_path: Optional[str] = None,
    prediction_params: Optional[str] = None,
    metrics: Optional[Dict[str, Callable[[List[Union[Tuple[str, str], Tuple[List[str], List[str]]]]], None]]] = None,
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

    # set metrics if none
    if metrics is None:
        metrics = get_default_metrics(task=model.task)

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

    # compute metrics
    # TODO: why should we have a metric_name?
    for metric_name, metric_f in metrics.items():
        metric_f([(sample.get_current_classification(), p) for sample, p in predicted_samples])


def main():
    args = parse_args()
    evaluate(
        model_checkpoint_path=args.model_checkpoint,
        cuda_device=args.cuda_device,
        token_batch_size=args.token_batch_size,
        input_path=args.f,
        output_path=args.o,
        prediction_params=args.prediction_params,
        metrics=None,
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
