from typing import List, Tuple, Union, Dict

from classy.data.data_drivers import SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample
from classy.pl_modules.mixins.task import TokensTask, SequenceTask, SentencePairTask, QATask, GenerationTask


class Evaluation:

    def __call__(
        self,
        predicted_samples: List[Tuple[Union[SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample], Union[str, List[str]]]],
    ) -> Dict:
        raise NotImplementedError
