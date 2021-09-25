from typing import Optional, List, Iterator, Tuple

import hydra
import omegaconf
import torch

from classy.models.consec.dataset import ConsecSample
from classy.models.consec.prediction import ConsecPredictionMixin
from classy.models.consec.sense_extractors import SenseExtractor
from classy.models.consec.task_ui import ConSeCTaskUIMixin
from classy.pl_modules.base import ClassyPLModule
from classy.pl_modules.mixins.task import TokensTask


class ConsecPLModule(ConSeCTaskUIMixin, ConsecPredictionMixin, TokensTask, ClassyPLModule):
    def __init__(
        self, sense_extractor: omegaconf.DictConfig, predictor: omegaconf.DictConfig, optim_conf: omegaconf.DictConfig
    ):
        super().__init__(vocabulary=None, optim_conf=optim_conf)
        self.save_hyperparameters(ignore="vocabulary")
        self.sense_extractor: SenseExtractor = hydra.utils.instantiate(sense_extractor)
        new_embedding_size = self.sense_extractor.model.config.vocab_size + 203
        self.sense_extractor.resize_token_embeddings(new_embedding_size)

        # instantiate predictor
        self.predictor = hydra.utils.instantiate(predictor)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        relative_positions: Optional[torch.Tensor] = None,
        definitions_mask: Optional[torch.Tensor] = None,
        gold_markers: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> dict:

        sense_extractor_output = self.sense_extractor.extract(
            input_ids, attention_mask, token_type_ids, relative_positions, definitions_mask, gold_markers
        )

        output_dict = {
            "pred_logits": sense_extractor_output.prediction_logits,
            "pred_probs": sense_extractor_output.prediction_probs,
            "pred_markers": sense_extractor_output.prediction_markers,
            "loss": sense_extractor_output.loss,
        }

        return output_dict

    def batch_predict(self, *args, **kwargs) -> Iterator[Tuple[ConsecSample, List[float]]]:
        batch_samples = kwargs.get("original_sample")
        batch_definitions_positions = kwargs.get("definitions_positions")

        batch_out = self.forward(*args, **kwargs)
        batch_predictions = batch_out["pred_probs"]

        for sample, dp, probs in zip(batch_samples, batch_definitions_positions, batch_predictions):
            yield sample, [probs[start].item() for start in dp]

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        forward_output = self.forward(**batch)
        self.log("loss", forward_output["loss"], on_step=False, on_epoch=True)
        return forward_output["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        forward_output = self.forward(**batch)
        self.log(f"val_loss", forward_output["loss"], prog_bar=True)
