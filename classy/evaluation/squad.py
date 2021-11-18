from typing import List, Tuple

from datasets import load_metric

from classy.data.data_drivers import QASample
from classy.evaluation.base import Evaluation


class SQuADV1Evaluation(Evaluation):
    def __init__(self):
        self.squad = load_metric("squad")

    def __call__(
        self,
        predicted_samples: List[Tuple[QASample, str]],
    ):

        pred = [
            {"id": sample.squad_id, "prediction_text": sample.context[start:end]}
            for sample, (start, end) in predicted_samples
        ]
        gold = [{"id": sample.squad_id, "answers": sample.full_answers} for sample, _ in predicted_samples]
        assert all(
            g["id"] is not None and g["answers"] is not None for g in gold
        ), f"Expected 'id' and 'answers' in gold, but found None"

        results = self.squad.compute(predictions=pred, references=gold)
        exact_match, f1 = results["exact_match"], results["f1"]

        return {"exact_match": exact_match, "f1": f1}
