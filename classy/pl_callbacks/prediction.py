import itertools
from pathlib import Path
from typing import Iterator, Tuple, Union, List, Dict, Any

import hydra
import pytorch_lightning as pl
import torchmetrics
from datasets import load_metric
from hydra._internal import instantiate
from omegaconf import DictConfig

from classy.data.data_drivers import SentencePairSample, SequenceSample, TokensSample, get_data_driver, TSV, TOKEN
from classy.pl_modules.base import ClassyPLModule
from classy.scripts.model.predict import predict
from classy.utils.log import get_project_logger


logger = get_project_logger(__name__)


class PredictionCallback:
    def __call__(
            self,
            name: str,
            predicted_samples: List[Tuple[Union[SentencePairSample, SequenceSample, TokensSample], Union[str, List[str]]]],
            model: ClassyPLModule,
            trainer: pl.Trainer,
    ):
        raise NotImplementedError


class FileDumperPredictionCallback(PredictionCallback):

    def __init__(self):
        self.folder = Path('file-dumped-prediction-callback')  # hydra takes care of dedicated path in the exp folder
        self.folder.mkdir()

    def __call__(
            self,
            name: str,
            predicted_samples: List[Tuple[Union[SentencePairSample, SequenceSample, TokensSample], Union[str, List[str]]]],
            model: ClassyPLModule,
            trainer: pl.Trainer,
    ):
        with open(str(self.folder.joinpath(f'{name}.{trainer.global_step}.tsv')), 'w') as f:
            for sample, prediction in predicted_samples:
                f.write(sample.pretty_print(classification_result=prediction) + '\n')


class SeqEvalPredictionCallback(PredictionCallback):

    def __init__(self):
        self.backend_metric = load_metric("seqeval")
        self.log_p_metric = torchmetrics.AverageMeter()
        self.log_r_metric = torchmetrics.AverageMeter()
        self.log_f1_metric = torchmetrics.AverageMeter()

    def __call__(
            self,
            name: str,
            predicted_samples: List[Tuple[TokensSample, List[str]]],
            model: ClassyPLModule,
            trainer: pl.Trainer,
    ):
        assert model.task == TOKEN and \
               isinstance(predicted_samples[0][0], TokensSample) and \
               isinstance(predicted_samples[0][1], list)
        metric_out = self.backend_metric.compute(
            predictions=[labels for _, labels in predicted_samples],
            references=[sample.labels for sample, _ in predicted_samples]
        )
        p, r, f1 = metric_out["overall_precision"], metric_out["overall_recall"], metric_out["overall_f1"]

        # todo lightning reset of metrics is yet unclear: fix once it becomes clear and delete the following block
        for metric in trainer._results.result_metrics:
            if metric.meta.name in [f"{name}_precision", f"{name}_recall", f"{name}_f1"]:
                metric.reset()

        model.log(f"{name}_precision", p, prog_bar=True, on_step=False, on_epoch=True)
        model.log(f"{name}_recall", r, prog_bar=True, on_step=False, on_epoch=True)
        model.log(f"{name}_f1", f1, prog_bar=True, on_step=False, on_epoch=True)

        logger.info(f'SeqEvalPredictionCallback with name {name} completed with scores (p={p:.4f}, r={r:.4f}, f1={f1:.4f})')


class PredictionPLCallback(pl.Callback):

    def __init__(
            self,
            prediction_confs: List[Dict[str, Any]],
            prediction_callbacks: Dict[str, DictConfig],
            prediction_dataset_conf: DictConfig,
    ):
        self.prediction_callbacks = {n: hydra.utils.instantiate(pc) for n, pc in prediction_callbacks.items()}
        self.prediction_dataset_conf = prediction_dataset_conf
        self.prediction_confs = []
        for prediction_conf in prediction_confs:
            self.prediction_confs.append((
                prediction_conf['name'],
                prediction_conf['path'],
                prediction_conf['token_batch_size'],
                prediction_conf['limit'],
                prediction_conf['enabled_prediction_callbacks'],
            ))



    def on_validation_epoch_start(self, trainer: pl.Trainer, model: ClassyPLModule) -> None:

        logger.info('Executing prediction callback')

        for (
            name,
            path,
            token_batch_size,
            limit,
            enabled_prediction_callbacks
        ) in self.prediction_confs:

            logger.info(f'Prediction callback processing configuration {name} with path={path}')

            extension = path.split('.')[-1]
            data_driver = get_data_driver(model.task, extension)

            with open(path) as f:
                lines_it = map(lambda l: l.strip(), f)
                # apply limits
                if trainer.global_step == 0:
                    # do only a dry run on first epoch (correspond to sanity check run)
                    lines_it = itertools.islice(lines_it, 5)
                elif limit != -1:
                    lines_it = itertools.islice(lines_it, limit)

                predicted_samples = list(
                    predict(
                        model,
                        lines_it,
                        data_driver=data_driver,
                        dataset_conf=self.prediction_dataset_conf,
                        token_batch_size=token_batch_size
                    )
                )

                for callback in enabled_prediction_callbacks:
                    logger.info(f'Executing inner callback {callback}')
                    self.prediction_callbacks[callback](name, predicted_samples, model, trainer)

            logger.info(f'Prediction callback finished processing configuration {name}')

        logger.info('Prediction callback completed')
