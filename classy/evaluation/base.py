from typing import List, Tuple, Union, Dict

from classy.data.data_drivers import SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample


class Evaluation:
    def __call__(
        self,
        predicted_samples: List[Union[SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample]],
    ) -> Dict:
        raise NotImplementedError
