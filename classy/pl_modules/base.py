from typing import NamedTuple, Optional, Union, List

import hydra
import omegaconf
import pytorch_lightning as pl
import torch

from classy.data.readers import SEQUENCE
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

    @property
    def task(self) -> str:
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> List[Union[str, List[str]]]:
        raise NotImplementedError

    def configure_optimizers(self):
        return hydra.utils.instantiate(self._optim_conf, _recursive_=False)(module=self)


class SequencePLModule(ClassyPLModule):

    @property
    def task(self) -> str:
        return SEQUENCE
