from typing import List, Tuple, Union, Dict

from classy.data.data_drivers import SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample


class Evaluation:
    def __call__(
        self,
        predicted_samples: List[
            Tuple[
                Union[SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample],
                Union[str, List[str], Tuple[int, int]],
            ]
        ],
    ) -> Dict:
        raise NotImplementedError
