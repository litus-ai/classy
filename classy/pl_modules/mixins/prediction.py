from typing import Union, Iterator, Dict

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities import move_data_to_device
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from classy.data.data_drivers import ClassySample


class PredictionMixin:
    """
    Simple Mixin to model the prediction behavior of a classy.pl_modules.base.ClassyPLModule.
    """

    def predict(
        self,
        samples: Iterator[ClassySample],
        dataset_conf: Union[Dict, DictConfig],
        token_batch_size: int = 1024,
        progress_bar: bool = False,
        **kwargs
    ) -> Iterator[ClassySample]:
        """
        Exposed method of each classy.pl_modules.base.ClassyPLModule invoked to annotate a collection of input
        samples.

        Args:
            samples: iterator over the samples that have to be annotated.
            dataset_conf: the dataset configuration used to instantiate the Dataset with hydra.
            token_batch_size: the maximum number of tokens in each batch.
            progress_bar: whether or not to show a progress bar of the prediction process.
            **kwargs: additional parameters. (Future proof atm)

        Returns:
            An iterator over the input samples with the predicted annotation updated.

        """

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
                    for sample in batch_out:
                        yield sample

        if progress_bar:
            iterator.close()

    def batch_predict(self, *args, **kwargs) -> Iterator[ClassySample]:
        """
        General method that must be implemented by each classy.pl_modules.base.ClassyPLModule in order to perform
        batch prediction.

        Returns:
            An iterator over a collection of samples with the predicted annotation updated with the model outputs.
        """
        raise NotImplementedError
