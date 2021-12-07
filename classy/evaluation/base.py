from typing import List, Dict

from classy.data.data_drivers import ClassySample


class Evaluation:
    def __call__(
        self,
        predicted_samples: List[ClassySample],
    ) -> Dict:
        raise NotImplementedError
