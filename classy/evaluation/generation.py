from typing import List, Tuple, Dict

import nltk
from datasets import load_metric

from classy.data.data_drivers import GenerationSample
from classy.evaluation.base import Evaluation
from classy.utils.log import get_project_logger

logger = get_project_logger(__name__)


class RougeEvaluation(Evaluation):
    def __init__(self):
        self.rouge = load_metric("rouge")

    def __call__(self, predicted_samples: List[Tuple[GenerationSample, str]]) -> Dict:
        assert all(sample.target_sequence is not None for sample, _ in predicted_samples)

        gold_summaries = [sample.target_sequence for sample, _ in predicted_samples]
        pred_summaries = [prediction for _, prediction in predicted_samples]

        # process summaries
        # todo maybe improve with something like ptb/stanza/some real sentence tokenizer
        gold_summaries = ["\n".join(nltk.sent_tokenize(gs.replace(". ", "\n").rstrip())) for gs in gold_summaries]
        pred_summaries = ["\n".join(nltk.sent_tokenize(ps.replace(". ", "\n").rstrip())) for ps in pred_summaries]

        results = self.rouge.compute(predictions=pred_summaries, references=gold_summaries)
        scores = {}

        for k, v in results.items():
            scores[k] = v.mid.fmeasure

        return scores


class SacreBleuEvaluation(Evaluation):
    def __init__(self):
        self.bleu = load_metric("sacrebleu")

    def __call__(
        self,
        predicted_samples: List[Tuple[GenerationSample, str]],
    ):

        assert all(sample.target_sequence is not None for sample, _ in predicted_samples)

        references = [sample.target_sequence for sample, _ in predicted_samples]
        predictions = [prediction for _, prediction in predicted_samples]

        results = self.bleu.compute(predictions=predictions, references=[[r] for r in references])
        return {"bleu": results["score"]}
