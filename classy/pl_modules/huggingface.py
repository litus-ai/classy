from typing import Any, NamedTuple, Optional

import hydra
import pytorch_lightning as pl
import torch
import torchmetrics
from transformers import AutoModelForSequenceClassification


class ClassificationOutput(NamedTuple):
    logits: torch.Tensor
    probabilities: torch.Tensor
    predictions: torch.Tensor
    loss: Optional[torch.Tensor] = None


class HFSequencePLModule(pl.LightningModule):

    def __init__(self, conf, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(self.hparams.model.transformer_model)
        self.accuracy = torchmetrics.Accuracy()
        self.precision = torchmetrics.Precision()
        self.recall = torchmetrics.Recall()
        self.f1 = torchmetrics.F1()

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None
    ) -> ClassificationOutput:
        model_input = {
            "input_ids": input_ids, "attention_mask": attention_mask
        }
        if token_type_ids is not None:
            model_input["token_type_ids"] = token_type_ids
        if labels is not None:
            model_input["labels"] = labels
        model_output = self.classifier(**model_input)
        return ClassificationOutput(
            logits=model_output.logits,
            probabilities=torch.softmax(model_output.logits, dim=-1),
            predictions=torch.argmax(model_output.logits, dim=-1),
            loss=model_output.loss
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        classification_output = self.forward(**batch)
        self.log("loss", classification_output.loss)
        return classification_output.loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        classification_output = self.forward(**batch)

        self.accuracy(classification_output.predictions, batch["labels"])
        self.precision(classification_output.predictions, batch["labels"])
        self.recall(classification_output.predictions, batch["labels"])
        self.f1(classification_output.predictions, batch["labels"])

        self.log("val_loss", classification_output.loss)
        self.log("accuracy", self.accuracy)
        self.log("precision", self.precision)
        self.log("recall", self.recall)
        self.log("f1-score", self.f1)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        classification_output = self.forward(**batch)

        self.accuracy(classification_output.predictions, batch["labels"])
        self.precision(classification_output.predictions, batch["labels"])
        self.recall(classification_output.predictions, batch["labels"])
        self.f1(classification_output.predictions, batch["labels"])

        self.log("accuracy", self.accuracy)
        self.log("precision", self.precision)
        self.log("recall", self.recall)
        self.log("f1-score", self.f1)

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.hparams.train.optim, _recursive_=False)(module=self)
