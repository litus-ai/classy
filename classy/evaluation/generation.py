from typing import Dict, List, Tuple

import nltk
from datasets import load_metric

from classy.data.data_drivers import GenerationSample
from classy.evaluation.base import Evaluation
from classy.utils.log import get_project_logger

logger = get_project_logger(__name__)


class RougeEvaluation(Evaluation):
    def __init__(self):
        self.rouge = load_metric("rouge")

    def __call__(self, path: str, predicted_samples: List[GenerationSample]) -> Dict:
        assert all(
            sample.reference_annotation is not None for sample in predicted_samples
        )

        gold_summaries = [sample.reference_annotation for sample in predicted_samples]
        pred_summaries = [sample.predicted_annotation for sample in predicted_samples]

        # process summaries
        # todo maybe improve with something like ptb/stanza/some real sentence tokenizer
        gold_summaries = [
            "\n".join(nltk.sent_tokenize(gs.replace(". ", "\n").rstrip()))
            for gs in gold_summaries
        ]
        pred_summaries = [
            "\n".join(nltk.sent_tokenize(ps.replace(". ", "\n").rstrip()))
            for ps in pred_summaries
        ]

        results = self.rouge.compute(
            predictions=pred_summaries, references=gold_summaries
        )
        scores = {}

        for k, v in results.items():
            scores[k] = v.mid.fmeasure

        return scores


class SacreBleuEvaluation(Evaluation):
    def __init__(self):
        self.bleu = load_metric("sacrebleu")

    def __call__(
        self,
        path: str,
        predicted_samples: List[GenerationSample],
    ):

        assert all(
            sample.reference_annotation is not None for sample in predicted_samples
        )

        references = [sample.reference_annotation for sample in predicted_samples]
        predictions = [sample.predicted_annotation for sample in predicted_samples]

        results = self.bleu.compute(
            predictions=predictions, references=[[r] for r in references]
        )
        return {"bleu": results["score"]}
