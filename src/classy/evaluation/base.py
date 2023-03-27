from typing import Dict, List

from ..data.data_drivers import ClassySample


class Evaluation:
    def __call__(
        self,
        path: str,
        predicted_samples: List[ClassySample],
    ) -> Dict:
        raise NotImplementedError
