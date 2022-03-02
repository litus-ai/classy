from typing import Dict, Iterator, Union

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
        dataset = hydra.utils.instantiate(
            dataset_conf, samples=samples_it(), vocabulary=self.vocabulary
        )

        # instantiate dataloader
        iterator = DataLoader(dataset, batch_size=None, num_workers=0)
        if progress_bar:
            iterator = tqdm(iterator, desc="Predicting")

        with torch.inference_mode():
            for batch in iterator:
                # do batch predict
                with autocast(enabled=True):  # todo: always enabled?
                    batch = move_data_to_device(batch, self.device)
                    batch_out = self.batch_predict(**batch)
                # update prediction position position
                for sample in batch_out:
                    position2predicted_sample[
                        sample._mixin_prediction_position
                    ] = sample
                # yield
                while next_prediction_position in position2predicted_sample:
                    yield position2predicted_sample[next_prediction_position]
                    del position2predicted_sample[next_prediction_position]
                    next_prediction_position += 1

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
