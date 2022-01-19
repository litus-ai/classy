import itertools
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from classy.data.data_drivers import ClassySample, get_data_driver
from classy.evaluation.base import Evaluation
from classy.pl_modules.base import ClassyPLModule
from classy.utils.log import get_project_logger

logger = get_project_logger(__name__)


class PredictionCallback:
    def __call__(
        self,
        name: str,
        path: str,
        predicted_samples: List[ClassySample],
        model: ClassyPLModule,
        trainer: pl.Trainer,
    ):
        raise NotImplementedError


class EvaluationPredictionCallback(PredictionCallback):
    def __init__(self, evaluation: Evaluation):
        self.evaluation = evaluation

    def __call__(
        self,
        name: str,
        path: str,
        predicted_samples: List[ClassySample],
        model: ClassyPLModule,
        trainer: pl.Trainer,
    ):
        logger.info(f"Starting evaluation {self.__class__.__name__} with name {name}")
        results = self.evaluation(path, predicted_samples)
        for k, v in results.items():
            model.log(f"{name}_{k}", v, prog_bar=True, on_step=False, on_epoch=True)
        str_results = ", ".join([f"{k}={v}" for k, v in results.items()])
        logger.info(
            f"Evaluation {self.__class__.__name__} with name {name} completed with results: ({str_results})"
        )


class FileDumperPredictionCallback(PredictionCallback):
    def __init__(self):
        self.folder = Path(
            "file-dumped-prediction-callback"
        )  # hydra takes care of dedicated path in the exp folder
        self.folder.mkdir(exist_ok=True)

    def __call__(
        self,
        name: str,
        path: str,
        predicted_samples: List[ClassySample],
        model: ClassyPLModule,
        trainer: pl.Trainer,
    ):
        with open(
            str(self.folder.joinpath(f"{name}.{trainer.global_step}.tsv")), "w"
        ) as f:
            for sample in predicted_samples:
                f.write(sample.pretty_print() + "\n")


class PredictionPLCallback(pl.Callback):
    def __init__(
        self,
        path: str,
        prediction_dataset_conf: DictConfig,
        on_result: Dict[str, DictConfig],
        settings: List[Dict[str, Any]],
    ):
        self.prediction_dataset_conf = prediction_dataset_conf
        self.on_result = {n: hydra.utils.instantiate(c) for n, c in on_result.items()}
        self.settings = []
        for prediction_conf in settings:
            self.settings.append(
                (
                    prediction_conf["name"],
                    prediction_conf["path"] or path,
                    prediction_conf["token_batch_size"],
                    prediction_conf.get("prediction_param_conf_path", None),
                    prediction_conf["limit"],
                    prediction_conf.get("on_result", list(self.on_result.keys())),
                )
            )

        self.slicing_cache = {}

    def _get_sliced_path(self, samples_it, data_driver, path, limit, extension):
        if (path, limit) not in self.slicing_cache:
            self.slicing_cache[(path, limit)] = tempfile.NamedTemporaryFile(
                suffix=f".{extension}"
            )
            data_driver.save(samples_it, self.slicing_cache[(path, limit)].name)
        return self.slicing_cache[(path, limit)].name

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, model: ClassyPLModule
    ) -> None:

        logger.info("Executing prediction callback")

        for (
            name,
            path,
            token_batch_size,
            prediction_param_conf_path,
            limit,
            on_result,
        ) in self.settings:

            logger.info(
                f"Prediction callback processing setting {name} with path={path}"
            )

            if prediction_param_conf_path is not None:
                model.load_prediction_params(
                    dict(OmegaConf.load(prediction_param_conf_path))
                )

            extension = path.split(".")[-1]
            data_driver = get_data_driver(model.task, extension)

            # build samples iterator
            samples_it = data_driver.read_from_path(path)

            # apply limits (changing path as well)
            if trainer.global_step == 0:
                # do only a dry run on first epoch (correspond to sanity check run)
                samples_it = itertools.islice(samples_it, 5)
                samples_it, saving_samples_it = itertools.tee(samples_it)
                path = self._get_sliced_path(
                    saving_samples_it,
                    data_driver,
                    path=path,
                    limit=5,
                    extension=extension,
                )
            elif limit != -1:
                # if provided, apply the limit given
                samples_it = itertools.islice(samples_it, limit)
                samples_it, saving_samples_it = itertools.tee(samples_it)
                path = self._get_sliced_path(
                    saving_samples_it,
                    data_driver,
                    path=path,
                    limit=limit,
                    extension=extension,
                )

            predicted_samples = list(
                model.predict(
                    samples_it,
                    dataset_conf=self.prediction_dataset_conf,
                    token_batch_size=token_batch_size,
                )
            )

            for callback in on_result:
                logger.info(f"Executing inner callback {callback}")
                self.on_result[callback](name, path, predicted_samples, model, trainer)

            logger.info(f"Prediction callback finished processing configuration {name}")

        logger.info("Prediction callback completed")
