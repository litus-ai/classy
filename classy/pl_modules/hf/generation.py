import re
from typing import Dict, Iterator, List, Optional, Tuple

import omegaconf
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from classy.data.data_drivers import GenerationSample
from classy.pl_modules.base import ClassificationOutput, ClassyPLModule
from classy.pl_modules.mixins.task import GenerationTask


class HFGenerationPLModule(GenerationTask, ClassyPLModule):
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """ """
        forward_output = self.forward(**batch)
        self.log("loss", forward_output.loss)
        self.log("ppl", torch.exp(forward_output.loss))
        return forward_output.loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """ """
        forward_output = self.forward(**batch)
        self.log("val_loss", forward_output.loss)
        self.log(
            "val_ppl",
            torch.exp(forward_output.loss),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return forward_output.loss

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """ """
        forward_output = self.forward(**batch)
        self.log("test_loss", forward_output.loss)
        self.log(
            "test_ppl",
            torch.exp(forward_output.loss),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return forward_output.loss


class BartGenerativeModule(HFGenerationPLModule):
    def __init__(
        self,
        transformer_model: str,
        decoding_skip_special_tokens: bool,
        decoding_clean_up_tokenization_spaces: bool,
        optim_conf: omegaconf.DictConfig,
        additional_special_tokens: Optional[List[str]] = None,
    ):
        super().__init__(vocabulary=None, optim_conf=optim_conf)
        self.tokenizer = AutoTokenizer.from_pretrained(
            transformer_model,
            additional_special_tokens=list(additional_special_tokens)
            if additional_special_tokens is not None
            else None,
            use_fast=True,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(transformer_model)
        if additional_special_tokens is not None and len(additional_special_tokens) > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.decoding_skip_special_tokens = decoding_skip_special_tokens
        self.decoding_clean_up_tokenization_spaces = (
            decoding_clean_up_tokenization_spaces
        )
        self.forced_bos_token_id = self.tokenizer.bos_token_id
        self.generation_params = {}

    def load_prediction_params(self, prediction_params: Dict):
        self.generation_params = prediction_params

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        decoder_attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> ClassificationOutput:
        bart_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        return ClassificationOutput(
            loss=bart_out.loss,
            logits=bart_out.logits,
            probabilities=bart_out.logits.softmax(dim=-1),
            predictions=bart_out.logits.argmax(dim=-1),
        )

    def batch_predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_start_token_id: torch.Tensor,
        **kwargs,
    ) -> Iterator[GenerationSample]:
        assert len(set(decoder_start_token_id.squeeze(-1).tolist())) == 1
        # generate
        bart_out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_start_token_id=decoder_start_token_id[0][0],
            forced_bos_token_id=self.forced_bos_token_id,
            **self.generation_params,
        )
        # decode
        decoded_bart_out = self.tokenizer.batch_decode(
            bart_out,
            skip_special_tokens=self.decoding_skip_special_tokens,
            clean_up_tokenization_spaces=self.decoding_clean_up_tokenization_spaces,
        )
        # handle num sequences
        num_sequences = int(len(decoded_bart_out) / input_ids.shape[0])
        grouped_decoded_bart_out = []
        for i in range(0, len(decoded_bart_out), num_sequences):
            grouped_decoded_bart_out.append(decoded_bart_out[i : i + num_sequences])
        # postprocess
        samples = kwargs.get("samples")
        for sample, prediction in zip(samples, grouped_decoded_bart_out):
            sample.predicted_annotation = prediction[0]
            if num_sequences > 1:
                sample.predicted_annotation_group = prediction
            yield sample


class MBartGenerativeModule(BartGenerativeModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forced_bos_token_id = None


class T5GenerativeModule(HFGenerationPLModule):
    def __init__(
        self,
        transformer_model: str,
        decoding_skip_special_tokens: bool,
        decoding_clean_up_tokenization_spaces: bool,
        optim_conf: omegaconf.DictConfig,
        additional_special_tokens: Optional[List[str]] = None,
    ):
        super().__init__(vocabulary=None, optim_conf=optim_conf)
        self.tokenizer = AutoTokenizer.from_pretrained(
            transformer_model,
            additional_special_tokens=list(additional_special_tokens)
            if additional_special_tokens is not None
            else None,
            use_fast=True,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(transformer_model)
        if additional_special_tokens is not None and len(additional_special_tokens) > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.decoding_skip_special_tokens = decoding_skip_special_tokens
        self.decoding_clean_up_tokenization_spaces = (
            decoding_clean_up_tokenization_spaces
        )
        self.generation_params = {}

    def load_prediction_params(self, prediction_params: Dict):
        self.generation_params = prediction_params

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        decoder_attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> ClassificationOutput:
        t5_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        return ClassificationOutput(
            loss=t5_out.loss,
            logits=t5_out.logits,
            probabilities=t5_out.logits.softmax(dim=-1),
            predictions=t5_out.logits.argmax(dim=-1),
        )

    def batch_predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> Iterator[GenerationSample]:
        # generate
        t5_out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **self.generation_params,
        )
        # decode
        decoded_t5_out = self.tokenizer.batch_decode(
            t5_out,
            skip_special_tokens=self.decoding_skip_special_tokens,
            clean_up_tokenization_spaces=self.decoding_clean_up_tokenization_spaces,
        )
        # handle num sequences
        num_sequences = int(len(decoded_t5_out) / input_ids.shape[0])
        grouped_decoded_t5_out = []
        for i in range(0, len(decoded_t5_out), num_sequences):
            grouped_decoded_t5_out.append(decoded_t5_out[i : i + num_sequences])
        # postprocess
        samples = kwargs.get("samples")
        for sample, prediction in zip(samples, grouped_decoded_t5_out):
            sample.predicted_annotation = prediction[0]
            if num_sequences > 1:
                sample.predicted_annotation_group = prediction
            yield sample


class GPT2GenerativeModule(HFGenerationPLModule):
    def __init__(
        self,
        transformer_model: str,
        decoding_skip_special_tokens: bool,
        decoding_clean_up_tokenization_spaces: bool,
        optim_conf: omegaconf.DictConfig,
        additional_special_tokens: Optional[List[str]] = None,
    ):
        super().__init__(vocabulary=None, optim_conf=optim_conf)
        self.tokenizer = AutoTokenizer.from_pretrained(
            transformer_model,
            additional_special_tokens=list(additional_special_tokens)
            if additional_special_tokens is not None
            else None,
            use_fast=True,
            add_prefix_space=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(transformer_model)
        if additional_special_tokens is not None and len(additional_special_tokens) > 0:
            self.model.model.shared = self.model.resize_token_embeddings(
                len(self.tokenizer)
            )
        self.decoding_skip_special_tokens = decoding_skip_special_tokens
        self.decoding_clean_up_tokenization_spaces = (
            decoding_clean_up_tokenization_spaces
        )
        self.generation_params = {}

    def load_prediction_params(self, prediction_params: Dict):
        self.generation_params = prediction_params

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        **kwargs,
    ) -> ClassificationOutput:
        gpt_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return ClassificationOutput(
            loss=gpt_out.loss,
            logits=gpt_out.logits,
            probabilities=gpt_out.logits.softmax(dim=-1),
            predictions=gpt_out.logits.argmax(dim=-1),
        )

    def batch_predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> Iterator[Tuple[GenerationSample, str]]:
        # generate
        gpt_out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **self.generation_params,
        )
        # decode
        decoded_gpt_out = self.tokenizer.batch_decode(
            gpt_out,
            skip_special_tokens=self.decoding_skip_special_tokens,
            clean_up_tokenization_spaces=self.decoding_clean_up_tokenization_spaces,
        )
        # handle num sequences
        num_sequences = int(len(decoded_gpt_out) / input_ids.shape[0])
        grouped_decoded_gpt_out = []
        for i in range(0, len(decoded_gpt_out), num_sequences):
            grouped_decoded_gpt_out.append(decoded_gpt_out[i : i + num_sequences])
        # postprocess
        samples = kwargs.get("samples")
        for sample, prediction in zip(samples, decoded_gpt_out):
            sample.predicted_annotation = prediction[0]
            if num_sequences > 1:
                sample.predicted_annotation_group = prediction
            yield sample
