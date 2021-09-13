from abc import ABC
from typing import Optional, List, Iterator, Tuple, Union

import omegaconf
import torch
import torchmetrics
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoModel, AutoConfig

from classy.data.data_drivers import SequenceSample, TokensSample, SentencePairSample
from classy.pl_modules.base import (
    ClassificationOutput,
    ClassyPLModule,
    TokensTask,
    SequenceTask,
    SentencePairTask,
)
from classy.utils.vocabulary import Vocabulary


# subclassed and mixed with both SequenceTask and SentencePairTask, as the underlying logic is identical (see below)
class HFSequenceCommonPLModule(ClassyPLModule, ABC):
    def __init__(
        self,
        transformer_model: str,
        vocabulary: Vocabulary,
        optim_conf: omegaconf.DictConfig,
    ):
        super().__init__(vocabulary=vocabulary, optim_conf=optim_conf)
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

    def predict(self, *args, **kwargs) -> Iterator[Tuple[Union[SequenceSample, SentencePairSample], str]]:
        samples = kwargs.get("samples")
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
        self.log("val_accuracy", self.accuracy_metric, prog_bar=True)
        self.log("val_precision", self.p_metric)
        self.log("val_recall", self.r_metric)
        self.log("val_f1-score", self.f1_metric, prog_bar=True)

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


class HFSequencePLModule(SequenceTask, HFSequenceCommonPLModule):
    pass


class HFSentencePairPLModule(SentencePairTask, HFSequenceCommonPLModule):
    pass


class HFTokensPLModule(TokensTask, ClassyPLModule):
    def __init__(
        self,
        transformer_model: str,
        use_last_n_layers: int,
        fine_tune: bool,
        vocabulary: Vocabulary,
        optim_conf: omegaconf.DictConfig,
    ):
        super().__init__(vocabulary=vocabulary, optim_conf=optim_conf)
        self.save_hyperparameters()

        # encoder
        auto_config = AutoConfig.from_pretrained(transformer_model)
        auto_config.output_hidden_states = True
        self.encoder = AutoModel.from_pretrained(transformer_model, config=auto_config)
        self.use_last_n_layers = use_last_n_layers

        if not fine_tune:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # classifier
        self.classification_head = nn.Linear(
            self.encoder.config.hidden_size, vocabulary.get_size(k="labels"), bias=False
        )
        self.criterion = torch.nn.CrossEntropyLoss()

        # metrics
        self.accuracy_metric = torchmetrics.Accuracy()
        self.p_metric = torchmetrics.Precision(mdmc_average="global")
        self.r_metric = torchmetrics.Recall(mdmc_average="global")
        self.f1_metric = torchmetrics.F1(mdmc_average="global")

        ignore_index = vocabulary.get_idx(k="labels", elem=Vocabulary.PAD)
        self.accuracy_metric = torchmetrics.Accuracy(
            mdmc_average="global",
            ignore_index=ignore_index,
        )
        self.p_metric = torchmetrics.Precision(
            mdmc_average="global",
            ignore_index=ignore_index,
        )
        self.r_metric = torchmetrics.Recall(
            mdmc_average="global",
            ignore_index=ignore_index,
        )
        self.f1_metric = torchmetrics.F1(
            mdmc_average="global",
            ignore_index=ignore_index,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_offsets: List[List[Tuple[int, int]]],
        samples: List[SequenceSample],
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> ClassificationOutput:

        # encode bpes and merge them (sum strategy)
        encoded_bpes = torch.stack(
            self.encoder(input_ids, attention_mask)[2][-self.use_last_n_layers :],
            dim=-1,
        ).sum(-1)

        # bpe -> token (first strategy)
        encoded_tokens = torch.zeros(
            (input_ids.shape[0], max(map(len, token_offsets)), encoded_bpes.shape[-1]),
            dtype=encoded_bpes.dtype,
            device=encoded_bpes.device,
        )
        for i, sample_offsets in enumerate(token_offsets):
            encoded_tokens[i, : len(sample_offsets)] = torch.stack([encoded_bpes[i, sj] for sj, ej in sample_offsets])

        # classify
        logits = self.classification_head(encoded_tokens)

        # return
        return ClassificationOutput(
            logits=logits,
            probabilities=logits.softmax(dim=-1),
            predictions=logits.argmax(dim=-1),
            loss=self.criterion(logits.view(-1, logits.shape[-1]), labels.view(-1)) if labels is not None else None,
        )

    def predict(self, *args, **kwargs) -> Iterator[Tuple[TokensSample, str]]:
        samples = kwargs.get("samples")
        classification_output = self.forward(*args, **kwargs)
        for sample, prediction in zip(samples, classification_output.predictions):
            yield sample, [
                self.vocabulary.get_elem(k="labels", idx=_p.item()) for _p in prediction[: len(sample.tokens)]
            ]

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        classification_output = self.forward(**batch)
        self.log("loss", classification_output.loss)
        return classification_output.loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        classification_output = self.forward(**batch)

        labels = batch["labels"].clone()
        labels[labels == -100] = self.vocabulary.get_idx(k="labels", elem=Vocabulary.PAD)

        self.accuracy_metric(classification_output.predictions, labels)
        self.p_metric(classification_output.predictions, labels)
        self.r_metric(classification_output.predictions, labels)
        self.f1_metric(classification_output.predictions, labels)

        self.log("val_loss", classification_output.loss)
        self.log("val_accuracy", self.accuracy_metric, prog_bar=True)
        self.log("val_precision", self.p_metric)
        self.log("val_recall", self.r_metric)
        self.log("val_f1-score", self.f1_metric, prog_bar=True)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        classification_output = self.forward(**batch)

        labels = batch["labels"].clone()
        labels[-100] = self.vocabulary.get_idx(k="labels", elem=Vocabulary.PAD)

        self.accuracy_metric(classification_output.predictions, labels)
        self.p_metric(classification_output.predictions, labels)
        self.r_metric(classification_output.predictions, labels)
        self.f1_metric(classification_output.predictions, labels)

        self.log("test_accuracy", self.accuracy_metric)
        self.log("test_precision", self.p_metric)
        self.log("test_recall", self.r_metric)
        self.log("test_f1-score", self.f1_metric)
