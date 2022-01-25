from typing import Dict, NamedTuple, Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch

from classy.pl_modules.mixins.prediction import PredictionMixin
from classy.pl_modules.mixins.saving import SavingMixin
from classy.utils.vocabulary import Vocabulary


class ClassificationOutput(NamedTuple):
    logits: torch.Tensor
    probabilities: torch.Tensor
    predictions: torch.Tensor
    loss: Optional[torch.Tensor] = None


class ClassyPLModule(SavingMixin, PredictionMixin, pl.LightningModule):
    def __init__(
        self, vocabulary: Optional[Vocabulary], optim_conf: omegaconf.DictConfig
    ):
        super().__init__()
        self.vocabulary: Vocabulary = vocabulary
        self._optim_conf = optim_conf

    def load_prediction_params(self, prediction_params: Dict):
        pass

    def forward(self, *args, **kwargs) -> ClassificationOutput:
        raise NotImplementedError

    def configure_optimizers(self):
        """ """
        return hydra.utils.instantiate(self._optim_conf, _recursive_=False)(module=self)
