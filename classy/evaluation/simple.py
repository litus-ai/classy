from typing import Dict, List, Tuple

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from classy.data.data_drivers import SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample
from classy.evaluation.base import Evaluation
from classy.utils.commons import flatten


def accuracy(gold, pred) -> float:
    return accuracy_score(gold, pred)


def p_r_f_support(gold, pred) -> Dict[str, float]:
    result = {}
    for avg in ["micro", "macro", "weighted"]:
        p, r, f1, _ = precision_recall_fscore_support(gold, pred, average=avg)
        for k, v in zip(["precision", "recall", "f1"], [p, r, f1]):
            result[f"{avg}_{k}"] = v
    return result


class SequenceSimpleEvaluation(Evaluation):
    def __call__(self, predicted_samples: List[Tuple[SequenceSample, str]]) -> Dict:
        gold, pred = [sample.get_current_classification() for sample, p in predicted_samples], [
            p for sample, p in predicted_samples
        ]
        return {"accuracy": accuracy(gold, pred), **p_r_f_support(gold, pred)}


class SentencePairSimpleEvaluation(Evaluation):
    def __call__(self, predicted_samples: List[Tuple[SentencePairSample, str]]) -> Dict:
        gold, pred = [sample.get_current_classification() for sample, p in predicted_samples], [
            p for sample, p in predicted_samples
        ]
        return {"accuracy": accuracy(gold, pred), **p_r_f_support(gold, pred)}


class TokenSimpleEvaluation(Evaluation):
    def __call__(self, predicted_samples: List[Tuple[TokensSample, List[str]]]) -> Dict:
        gold = [sample.get_current_classification() for sample, p in predicted_samples]
        pred = [p for sample, p in predicted_samples]
        gold, pred = flatten(gold), flatten(pred)
        return {"accuracy": accuracy(gold, pred), **p_r_f_support(gold, pred)}


class QASimpleEvaluation(Evaluation):
    """
    Computes a simple exact-match accuracy
    """

    def __call__(self, predicted_samples: List[Tuple[QASample, Tuple[int, int]]]) -> Dict:
        n, d = 0, 0
        for sample, p in predicted_samples:
            d += 1
            if sample.get_current_classification() == p:
                n += 1
        return {"exact-match-accuracy": f"{n / d:.2f}"}


class GenerationSimpleEvaluation(Evaluation):
    """
    Computes a simple full-text accuracy
    """

    def __call__(self, predicted_samples: List[Tuple[GenerationSample, str]]) -> Dict:
        n, d = 0, 0
        for sample, p in predicted_samples:
            d += 1
            if sample.get_current_classification() == p:
                n += 1
        return {"full-generation-accuracy": f"{n / d:.2f}"}
