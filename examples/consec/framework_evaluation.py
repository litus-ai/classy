import argparse
import tempfile
from typing import Tuple, List

import pytorch_lightning as pl
import torch

from classy.data.data_drivers import TokensSample, get_data_driver
from classy.pl_callbacks.prediction import PredictionCallback
from classy.pl_modules.base import ClassyPLModule
from classy.utils.commons import execute_bash_command
from classy.utils.lightning import load_classy_module_from_checkpoint, load_prediction_dataset_conf_from_checkpoint
from classy.utils.log import get_project_logger

logger = get_project_logger(__name__)


def framework_evaluate(
    raganato_scorer_folder: str, gold_file_path: str, pred_file_path: str
) -> Tuple[float, float, float]:
    command_output = execute_bash_command(
        f"[ ! -e {raganato_scorer_folder}/Scorer.class ] && javac -d {raganato_scorer_folder} {raganato_scorer_folder}/Scorer.java; java -cp {raganato_scorer_folder} Scorer {gold_file_path} {pred_file_path}"
    )
    command_output = command_output.split("\n")
    p, r, f1 = [float(command_output[i].split("=")[-1].strip()[:-1]) for i in range(3)]
    return p, r, f1


class RaganatoPredictionCallback(PredictionCallback):
    def __init__(self, gold_key_path: str, scorer_folder: str):
        self.gold_key_path = gold_key_path
        self.scorer_folder = scorer_folder

    def __call__(
        self,
        name: str,
        predicted_samples: List[Tuple[TokensSample, List[str]]],
        model: ClassyPLModule,
        trainer: pl.Trainer,
    ):
        # write predictions and evaluate
        with tempfile.TemporaryDirectory() as tmp_dir:

            # write predictions to tmp file
            with open(f"{tmp_dir}/predictions.gold.key.txt", "w") as f:
                for token_sample, labels in predicted_samples:
                    for instance_id, label in zip(token_sample.instance_ids, labels):
                        f.write(f"{instance_id} {label}\n")

            # compute metrics
            p, r, f1 = framework_evaluate(
                self.scorer_folder,
                gold_file_path=self.gold_key_path,
                pred_file_path=f"{tmp_dir}/predictions.gold.key.txt",
            )

            # todo lightning reset of metrics is yet unclear: fix once it becomes clear and delete the following block
            for metric in trainer._results.result_metrics:
                if metric.meta.name == f"{name}_f1":
                    metric.reset()

            logger.info(f"Raganato callback {name} finished with f1 score: {f1:.4f}")
            model.log(f"{name}_f1", f1, prog_bar=True, on_step=False, on_epoch=True)


def main(
    model_checkpoint_path: str,
    input_path: str,
    gold_path: str,
    scorer_folder: str,
    cuda_device: int,
    token_batch_size: int,
):

    model = load_classy_module_from_checkpoint(model_checkpoint_path)
    model.to(torch.device(cuda_device if cuda_device != -1 else "cpu"))
    model.freeze()

    dataset_conf = load_prediction_dataset_conf_from_checkpoint(model_checkpoint_path)
    input_extension = input_path.split(".")[-1]
    data_driver = get_data_driver(model.task, input_extension)

    with open("/tmp/consec.txt", "w") as f:
        for token_sample, labels in model.predict(
            data_driver.read_from_path(input_path),
            token_batch_size=token_batch_size,
            dataset_conf=dataset_conf,
            progress_bar=True,
        ):
            for instance_id, label in zip(token_sample.instance_ids, labels):
                f.write(f"{instance_id} {label}\n")

    # compute metrics
    p, r, f1 = framework_evaluate(
        scorer_folder,
        gold_file_path=gold_path,
        pred_file_path="/tmp/consec.txt",
    )

    print(f"# p: {p:.4f}")
    print(f"# r: {r:.4f}")
    print(f"# f1: {f1:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_checkpoint", type=str, help="Path to pl_modules checkpoint")
    parser.add_argument("--input-path", type=str, default=None, help="Input file (jsonl)")
    parser.add_argument("--gold-path", type=str, default=None, help="Gold key file")
    parser.add_argument("--scorer-folder", type=str, default=None, help="Scorer folder")
    parser.add_argument("--cuda-device", type=int, default=-1, help="Cuda device")
    parser.add_argument("--token-batch-size", type=int, default=1000, help="Token batch size")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        model_checkpoint_path=args.model_checkpoint,
        input_path=args.input_path,
        gold_path=args.gold_path,
        scorer_folder=args.scorer_folder,
        cuda_device=args.cuda_device,
        token_batch_size=args.token_batch_size,
    )
