from typing import Union, List, Iterator, Tuple, Dict, Any, Generator

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities import move_data_to_device
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from classy.data.data_drivers import (
    SentencePairSample,
    SequenceSample,
    TokensSample,
    QASample,
    GenerationSample,
)


class PredictionMixin:
    def predict(
        self,
        samples: Iterator[Union[SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample]],
        dataset_conf: Union[Dict, DictConfig],
        token_batch_size: int = 1024,
        progress_bar: bool = False,
        **kwargs
    ) -> Generator[
        Tuple[
            Union[SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample],
            Union[str, List[str], Tuple[int, int]],
        ],
        None,
        None,
    ]:

        # setup infrastructure to re-yield in order
        def samples_it():
            for i, sample in enumerate(samples):
                assert sample._mixin_prediction_position is None
                sample._mixin_prediction_position = i
                yield sample
        next_prediction_position = 0
        position2predicted_sample = {}

        # instantiate dataset
        dataset_conf["tokens_per_batch"] = token_batch_size
        dataset = hydra.utils.instantiate(dataset_conf, samples=samples_it(), vocabulary=self.vocabulary)

        # instantiate dataloader
        iterator = DataLoader(dataset, batch_size=None, num_workers=0)
        if progress_bar:
            iterator = tqdm(iterator, desc="Predicting")

        for batch in iterator:
            with autocast(enabled=True):  # todo: always enabled?
                with torch.inference_mode():
                    # do batch predict
                    batch = move_data_to_device(batch, self.device)
                    batch_out = self.batch_predict(**batch)
                    # update prediction position position
                    for sample, prediction in batch_out:
                        position2predicted_sample[sample._mixin_prediction_position] = (sample, prediction)
                    # yield
                    while next_prediction_position in position2predicted_sample:
                        yield position2predicted_sample[next_prediction_position]
                        del position2predicted_sample[next_prediction_position]
                        next_prediction_position += 1

        if progress_bar:
            iterator.close()

    def batch_predict(self, *args, **kwargs) -> Iterator[Tuple[Any, Any]]:
        raise NotImplementedError
