import itertools
from pathlib import Path
from typing import Tuple, Union, List, Dict, Any

import hydra
import nltk
import pytorch_lightning as pl
from datasets import load_metric
from omegaconf import DictConfig, OmegaConf

from classy.data.data_drivers import (
    SentencePairSample,
    SequenceSample,
    TokensSample,
    get_data_driver,
    TOKEN,
    GenerationSample,
    QASample,
)
from classy.pl_modules.base import ClassyPLModule
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
        self.folder = Path("file-dumped-prediction-callback")  # hydra takes care of dedicated path in the exp folder
        self.folder.mkdir(exist_ok=True)

    def __call__(
        self,
        name: str,
        predicted_samples: List[Tuple[Union[SentencePairSample, SequenceSample, TokensSample], Union[str, List[str]]]],
        model: ClassyPLModule,
        trainer: pl.Trainer,
    ):
        with open(str(self.folder.joinpath(f"{name}.{trainer.global_step}.tsv")), "w") as f:
            for sample, prediction in predicted_samples:
                f.write(sample.pretty_print(classification_result=prediction) + "\n")


class SeqEvalPredictionCallback(PredictionCallback):
    def __init__(self):
        self.backend_metric = load_metric("seqeval")

    def __call__(
        self,
        name: str,
        predicted_samples: List[Tuple[TokensSample, List[str]]],
        model: ClassyPLModule,
        trainer: pl.Trainer,
    ):
        assert (
            model.task == TOKEN
            and isinstance(predicted_samples[0][0], TokensSample)
            and isinstance(predicted_samples[0][1], list)
        )
        metric_out = self.backend_metric.compute(
            predictions=[labels for _, labels in predicted_samples],
            references=[sample.labels for sample, _ in predicted_samples],
        )
        p, r, f1 = metric_out["overall_precision"], metric_out["overall_recall"], metric_out["overall_f1"]

        model.log(f"{name}_precision", p, prog_bar=True, on_step=False, on_epoch=True)
        model.log(f"{name}_recall", r, prog_bar=True, on_step=False, on_epoch=True)
        model.log(f"{name}_f1", f1, prog_bar=True, on_step=False, on_epoch=True)

        logger.info(
            f"SeqEvalPredictionCallback with name {name} completed with scores (p={p:.4f}, r={r:.4f}, f1={f1:.4f})"
        )


class SummarizationRougeGenerationCallback(PredictionCallback):
    def __init__(self):
        self.rouge = load_metric("rouge")

    def __call__(
        self,
        name: str,
        predicted_samples: List[Tuple[GenerationSample, str]],
        model: ClassyPLModule,
        trainer: pl.Trainer,
    ):

        assert all(sample.target_sequence is not None for sample, _ in predicted_samples)

        gold_summaries = [sample.target_sequence for sample, _ in predicted_samples]
        pred_summaries = [prediction for _, prediction in predicted_samples]

        # process summaries
        # todo maybe improve with something like ptb/stanza/some real sentence tokenizer
        gold_summaries = ["\n".join(nltk.sent_tokenize(gs.replace(". ", "\n").rstrip())) for gs in gold_summaries]
        pred_summaries = ["\n".join(nltk.sent_tokenize(ps.replace(". ", "\n").rstrip())) for ps in pred_summaries]

        results = self.rouge.compute(predictions=pred_summaries, references=gold_summaries)
        scores = []

        for k, v in results.items():
            model.log(f"{name}_{k}", v.mid.fmeasure, prog_bar=True, on_step=False, on_epoch=True)
            scores.append(f"{name}_{k}: {v.mid.fmeasure:.4f}")

        logger.info(f"SummarizationRougeGenerationCallback with name {name} completed with scores ({','.join(scores)})")


class SacreBleuGenerationCallback(PredictionCallback):
    def __init__(self):
        self.bleu = load_metric("sacrebleu")

    def __call__(
        self,
        name: str,
        predicted_samples: List[Tuple[GenerationSample, str]],
        model: ClassyPLModule,
        trainer: pl.Trainer,
    ):

        assert all(sample.target_sequence is not None for sample, _ in predicted_samples)

        references = [sample.target_sequence for sample, _ in predicted_samples]
        predictions = [prediction for _, prediction in predicted_samples]

        results = self.bleu.compute(predictions=predictions, references=[[r] for r in references])
        score = results["score"]

        model.log(f"{name}_bleu", score, prog_bar=True, on_step=False, on_epoch=True)
        logger.info(f"SacreBleuGenerationCallback with name {name} completed with score: {score:.2f}")


class SQuADV1Callback(PredictionCallback):
    def __init__(self):
        self.squad = load_metric("squad")

    def __call__(
        self,
        name: str,
        predicted_samples: List[Tuple[QASample, str]],
        model: ClassyPLModule,
        trainer: pl.Trainer,
    ):

        pred = [
            {"id": sample.squad_id, "prediction_text": sample.context[start:end]}
            for sample, (start, end) in predicted_samples
        ]
        gold = [{"id": sample.squad_id, "answers": sample.full_answers} for sample, _ in predicted_samples]

        results = self.squad.compute(predictions=pred, references=gold)
        exact_match, f1 = results["exact_match"], results["f1"]

        model.log(f"{name}_exact_match", exact_match, prog_bar=True, on_step=False, on_epoch=True)
        model.log(f"{name}_f1", f1, prog_bar=True, on_step=False, on_epoch=True)

        logger.info(f"SQuADCallback with name {name} completed with score (exact_match={exact_match:.2f}, f1={f1:.2f})")


class PredictionPLCallback(pl.Callback):
    def __init__(
        self,
        validation_path: str,
        prediction_confs: List[Dict[str, Any]],
        prediction_callbacks: Dict[str, DictConfig],
        prediction_dataset_conf: DictConfig,
    ):
        self.prediction_callbacks = {n: hydra.utils.instantiate(pc) for n, pc in prediction_callbacks.items()}
        self.prediction_dataset_conf = prediction_dataset_conf
        self.prediction_confs = []
        for prediction_conf in prediction_confs:
            self.prediction_confs.append(
                (
                    prediction_conf["name"],
                    prediction_conf["path"] or validation_path,
                    prediction_conf["token_batch_size"],
                    prediction_conf.get("prediction_param_conf_path", None),
                    prediction_conf["limit"],
                    prediction_conf["enabled_prediction_callbacks"],
                )
            )

    def on_validation_epoch_start(self, trainer: pl.Trainer, model: ClassyPLModule) -> None:

        logger.info("Executing prediction callback")

        for (
            name,
            path,
            token_batch_size,
            prediction_param_conf_path,
            limit,
            enabled_prediction_callbacks,
        ) in self.prediction_confs:

            logger.info(f"Prediction callback processing configuration {name} with path={path}")

            if prediction_param_conf_path is not None:
                model.load_prediction_params(dict(OmegaConf.load(prediction_param_conf_path)))

            extension = path.split(".")[-1]
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
                    model.predict(
                        data_driver.read(lines_it),
                        dataset_conf=self.prediction_dataset_conf,
                        token_batch_size=token_batch_size,
                    )
                )

                for callback in enabled_prediction_callbacks:
                    logger.info(f"Executing inner callback {callback}")
                    self.prediction_callbacks[callback](name, predicted_samples, model, trainer)

            logger.info(f"Prediction callback finished processing configuration {name}")

        logger.info("Prediction callback completed")
