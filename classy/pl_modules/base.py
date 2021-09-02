import collections
from typing import NamedTuple, Optional, Union, List, Type, Iterator, Tuple, Dict

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torchmetrics

from classy.data.dataset.base import BaseDataset
from classy.data.data_drivers import SEQUENCE, SentencePairSample, SequenceSample, TokensSample, TOKEN
from classy.utils.vocabulary import Vocabulary


class ClassificationOutput(NamedTuple):
    logits: torch.Tensor
    probabilities: torch.Tensor
    predictions: torch.Tensor
    loss: Optional[torch.Tensor] = None


class ClassyPLModule(pl.LightningModule):
    def __init__(self, vocabulary: Vocabulary, optim_conf: omegaconf.DictConfig):
        super().__init__()
        self.vocabulary: Vocabulary = vocabulary
        self._optim_conf = optim_conf
        self.custom_metric_on_validation_end = collections.defaultdict(lambda: torchmetrics.AverageMeter())

    @property
    def task(self) -> str:
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> List[Iterator[Tuple[Union[SentencePairSample, SequenceSample, TokensSample], Union[str, List[str]]]]]:
        raise NotImplementedError

    def configure_optimizers(self):
        return hydra.utils.instantiate(self._optim_conf, _recursive_=False)(module=self)

    def log_custom_metric_on_validation_end(self, k, v):
        metric = self.custom_metric_on_validation_end[k]
        metric.reset()
        metric(v)



class SequencePLModule(ClassyPLModule):

    @property
    def task(self) -> str:
        return SEQUENCE

    def predict(self, *args, **kwargs) -> List[Iterator[Tuple[SequenceSample, str]]]:
        raise NotImplementedError


class TokensPLModule(ClassyPLModule):

    @property
    def task(self) -> str:
        return TOKEN

    def predict(self, *args, **kwargs) -> List[Iterator[Tuple[TokensSample, str]]]:
        raise NotImplementedError


