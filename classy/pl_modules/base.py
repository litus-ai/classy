import collections
from typing import Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torchmetrics

from classy.pl_modules.mixins.prediction import PredictionMixin
from classy.pl_modules.mixins.saving import SavingMixin
from classy.utils.vocabulary import Vocabulary


class ClassyPLModule(SavingMixin, PredictionMixin, pl.LightningModule):
    def __init__(self, vocabulary: Optional[Vocabulary], optim_conf: omegaconf.DictConfig):
        super().__init__()
        self.vocabulary: Vocabulary = vocabulary
        self._optim_conf = optim_conf

    def configure_optimizers(self):
        return hydra.utils.instantiate(self._optim_conf, _recursive_=False)(module=self)
