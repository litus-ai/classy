from typing import List, Optional

import hydra
import torch
from omegaconf import DictConfig

from classy.optim.optimizers.radam import RAdam


class Factory:
    """Factory interface that allows for simple instantiation of optimizers and schedulers for PyTorch Lightning.
    This class is essentially a work-around for lazy instantiation:
    * all params but for the module to be optimized are received in __init__
    * the actual instantiation of optimizers and schedulers takes place in the __call__ method, where the module to be
      optimized is provided
    __call__ will be invoked in the configure_optimizers hooks of LighiningModule-s and its return object directly returned.
    As such, the return type of __call__ can be any of those allowed by configure_optimizers, namely:
    * Single optimizer
    * List or Tuple - List of optimizers
    * Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict)
    * Dictionary, with an ‘optimizer’ key, and (optionally) a ‘lr_scheduler’ key whose value is a single LR scheduler or lr_dict
    * Tuple of dictionaries as described, with an optional ‘frequency’ key
    * None - Fit will run without any optimizer
    """

    def __call__(self, module: torch.nn.Module):
        raise NotImplementedError


class TorchFactory(Factory):
    """Simple factory wrapping standard PyTorch optimizers and schedulers."""

    # todo add scheduler support as well

    def __init__(self, optimizer: DictConfig):
        self.optimizer = optimizer

    def __call__(self, module: torch.nn.Module):
        return hydra.utils.instantiate(self.optimizer, params=module.parameters())


class RAdamWithDecayFactory(Factory):
    """Factory for RAdam optimizer."""

    def __init__(self, lr: float, weight_decay: float, no_decay_params: Optional[List[str]]):
        self.lr = lr
        self.weight_decay = weight_decay
        self.no_decay_params = no_decay_params

    def __call__(self, module: torch.nn.Module):

        if self.no_decay_params is not None:

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in module.named_parameters() if not any(nd in n for nd in self.no_decay_params)
                    ],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [p for n, p in module.named_parameters() if any(nd in n for nd in self.no_decay_params)],
                    "weight_decay": 0.0,
                },
            ]

        else:

            optimizer_grouped_parameters = [{"params": module.parameters(), "weight_decay": self.weight_decay}]

        optimizer = RAdam(optimizer_grouped_parameters, lr=self.lr, weight_decay=self.weight_decay)

        return optimizer
