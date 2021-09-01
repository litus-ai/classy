from typing import Any, NamedTuple, Optional, Dict, List, Union, Iterator, Tuple

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torchmetrics
from transformers import AutoModelForSequenceClassification

from classy.data.data_drivers import SequenceSample
from classy.pl_modules.base import SequencePLModule
from classy.utils.vocabulary import Vocabulary


class ClassificationOutput(NamedTuple):
    logits: torch.Tensor
    probabilities: torch.Tensor
    predictions: torch.Tensor
    loss: Optional[torch.Tensor] = None


class HFSequencePLModule(SequencePLModule):
    def __init__(self, transformer_model: str, optim_conf: omegaconf.DictConfig, vocabulary: Vocabulary):
        super().__init__(vocabulary, optim_conf)
        self.save_hyperparameters()
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            transformer_model, num_labels=vocabulary.get_size(k="labels")
        )
        self.accuracy_metric = torchmetrics.Accuracy()
        self.p_metric = torchmetrics.Precision()
        self.r_metric = torchmetrics.Recall()
        self.f1_metric = torchmetrics.F1()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        samples: List[SequenceSample],
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> ClassificationOutput:
        model_input = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            model_input["token_type_ids"] = token_type_ids
        if labels is not None:
            model_input["labels"] = labels
        model_output = self.classifier(**model_input)
        return ClassificationOutput(
            logits=model_output.logits,
            probabilities=torch.softmax(model_output.logits, dim=-1),
            predictions=torch.argmax(model_output.logits, dim=-1),
            loss=model_output.loss,
        )

    def predict(self, *args, **kwargs) -> Iterator[Tuple[SequenceSample, str]]:
        samples = kwargs.get('samples')
        classification_output = self.forward(*args, **kwargs)
        for sample, prediction in zip(samples, classification_output.predictions):
            yield sample, self.vocabulary.get_elem(k="labels", idx=prediction.item())

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        classification_output = self.forward(**batch)
        self.log("loss", classification_output.loss)
        return classification_output.loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        classification_output = self.forward(**batch)

        self.accuracy_metric(classification_output.predictions, batch["labels"].squeeze(-1))
        self.p_metric(classification_output.predictions, batch["labels"].squeeze(-1))
        self.r_metric(classification_output.predictions, batch["labels"].squeeze(-1))
        self.f1_metric(classification_output.predictions, batch["labels"].squeeze(-1))

        self.log("val_loss", classification_output.loss)
        self.log("val_accuracy", self.accuracy_metric)
        self.log("val_precision", self.p_metric)
        self.log("val_recall", self.r_metric)
        self.log("val_f1-score", self.f1_metric)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        classification_output = self.forward(**batch)

        self.accuracy_metric(classification_output.predictions, batch["labels"].squeeze(-1))
        self.p_metric(classification_output.predictions, batch["labels"].squeeze(-1))
        self.r_metric(classification_output.predictions, batch["labels"].squeeze(-1))
        self.f1_metric(classification_output.predictions, batch["labels"].squeeze(-1))

        self.log("test_accuracy", self.accuracy_metric)
        self.log("test_precision", self.p_metric)
        self.log("test_recall", self.r_metric)
        self.log("test_f1-score", self.f1_metric)
