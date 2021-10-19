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
    QASample, GenerationSample,
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

        # instantiate dataset
        dataset_conf["tokens_per_batch"] = token_batch_size
        dataset = hydra.utils.instantiate(dataset_conf, samples=samples, vocabulary=self.vocabulary)

        # instantiate dataloader
        iterator = DataLoader(dataset, batch_size=None, num_workers=0)
        if progress_bar:
            iterator = tqdm(iterator, desc="Predicting")

        for batch in iterator:
            with autocast(enabled=True):  # todo: always enabled?
                with torch.inference_mode():
                    batch = move_data_to_device(batch, self.device)
                    batch_out = self.batch_predict(**batch)
                    for sample, prediction in batch_out:
                        yield sample, prediction

        if progress_bar:
            iterator.close()

    def batch_predict(self, *args, **kwargs) -> Iterator[Tuple[Any, Any]]:
        raise NotImplementedError
