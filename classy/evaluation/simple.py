from typing import Dict, List, Tuple, Union

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from classy.data.data_drivers import SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample
from classy.evaluation.base import Evaluation


def accuracy(gold, pred) -> float:
    return accuracy_score(gold, pred)


def p_r_f_support(gold, pred) -> Dict[str, float]:
    result = {}
    for avg in ["micro", "macro", "weighted"]:
        p, r, f1, _ = precision_recall_fscore_support(gold, pred, average=avg)
        for k, v in zip(["precision", "recall", "f1"], [p, r, f1]):
            result[f'{avg}_{k}'] = v
    return result


class SequenceSimpleEvaluation(Evaluation):

    def __call__(
            self,
            predicted_samples: List[Tuple[SequenceSample, str]]
    ) -> Dict:
        gold, pred = [sample.get_current_classification() for sample, p in predicted_samples], [p for sample, p in predicted_samples]
        return {
            "accuracy": accuracy(gold, pred),
            **p_r_f_support(gold, pred)
        }


class SentencePairSimpleEvaluation(Evaluation):

    def __call__(
            self,
            predicted_samples: List[Tuple[SentencePairSample, str]]
    ) -> Dict:
        gold, pred = [sample.get_current_classification() for sample, p in predicted_samples], [p for sample, p in predicted_samples]
        return {
            "accuracy": accuracy(gold, pred),
            **p_r_f_support(gold, pred)
        }


class TokenSimpleEvaluation(Evaluation):

    def __call__(
            self,
            predicted_samples: List[Tuple[TokensSample, str]]
    ) -> Dict:
        gold, pred = [sample.get_current_classification() for sample, p in predicted_samples], [p for sample, p in predicted_samples]
        gold, pred = flatten(gold), flatten(pred)
        return {
            "accuracy": accuracy(gold, pred),
            **p_r_f_support(gold, pred)
        }


class QASimpleEvaluation(Evaluation):

    def __call__(
            self,
            predicted_samples: List[Tuple[QASample, str]]
    ) -> Dict:
        raise NotImplementedError


class GenerationSimpleEvaluation(Evaluation):

    def __call__(
            self,
            predicted_samples: List[Tuple[GenerationSimpleEvaluation, str]]
    ) -> Dict:
        raise NotImplementedError