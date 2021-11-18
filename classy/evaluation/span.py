from typing import List, Tuple, Dict

from datasets import load_metric

from classy.data.data_drivers import TokensSample
from classy.evaluation.base import Evaluation


class SeqEvalSpanEvaluation(Evaluation):
    def __init__(self):
        self.backend_metric = load_metric("seqeval")

    def __call__(
        self,
        predicted_samples: List[Tuple[TokensSample, List[str]]],
    ) -> Dict:

        metric_out = self.backend_metric.compute(
            predictions=[labels for _, labels in predicted_samples],
            references=[sample.labels for sample, _ in predicted_samples],
        )
        p, r, f1 = metric_out["overall_precision"], metric_out["overall_recall"], metric_out["overall_f1"]

        return {"precision": p, "recall": r, "f1": f1}
