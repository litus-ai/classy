import copy
import re

import hydra
from omegaconf import DictConfig, OmegaConf


def adapt_dataset_from(training_dataset: DictConfig, setting: str):
    # duplicate configuration
    training_dataset = copy.deepcopy(training_dataset)
    # resolve otherwise hydra.utils.instantiate will crash
    OmegaConf.resolve(training_dataset)
    # identify adaptation code
    train_dataset_class = training_dataset["_target_"]
    if not train_dataset_class.split(".")[-1][
        0
    ].isupper():  # if it is not upper then it is a class method
        train_dataset_class = ".".join(train_dataset_class.split(".")[:-1])
    # invoke it and return
    return hydra.utils.instantiate(
        {"_target_": f"{train_dataset_class}.adapt_dataset_from"},
        training_dataset=training_dataset,
        setting=setting,
        _recursive_=False,
    )


OmegaConf.register_new_resolver("adapt_dataset_from", adapt_dataset_from)


def resolve_hf_generation_base_dataset_on_transformer_model(
    transformer_model: str,
) -> str:
    if re.fullmatch("facebook/bart-(base|large)", transformer_model):
        return "classy.data.dataset.hf.generation.BartHFGenerationDataset.from_file"
    elif re.fullmatch("facebook/mbart-large-(cc25|50)", transformer_model):
        return "classy.data.dataset.hf.generation.MBartHFGenerationDataset.from_file"
    elif transformer_model.startswith("gpt2"):
        return "classy.data.dataset.hf.generation.GPT2HFGenerationCataset.from_file"
    elif (
        transformer_model.startswith("t5-")
        or transformer_model.startswith("google/t5-")
        or transformer_model.startswith("google/mt5-")
    ):
        return "classy.data.dataset.hf.generation.T5HFGenerationDataset.from_file"
    else:
        raise ValueError(
            f"{transformer_model} not currently supported in automatic resolution. But you can still write your own dataset (write _target_ and its parameters)."
        )


OmegaConf.register_new_resolver(
    "resolve_hf_generation_base_dataset_on_transformer_model",
    resolve_hf_generation_base_dataset_on_transformer_model,
)


def resolve_hf_generation_module_on_transformer_model(
    transformer_model: str,
) -> str:
    if re.fullmatch("facebook/bart-(base|large)", transformer_model):
        return "classy.pl_modules.hf.generation.BartGenerativeModule"
    elif re.fullmatch("facebook/mbart-large-(cc25|50)", transformer_model):
        return "classy.pl_modules.hf.generation.MBartGenerativeModule"
    elif transformer_model.startswith("gpt2"):
        return "classy.pl_modules.hf.generation.GPT2GenerativeModule"
    elif (
        transformer_model.startswith("t5-")
        or transformer_model.startswith("google/t5-")
        or transformer_model.startswith("google/mt5-")
    ):
        return "classy.pl_modules.hf.generation.T5GenerativeModule"
    else:
        raise ValueError(
            f"{transformer_model} not currently supported in automatic resolution. But you can still write your own dataset (write _target_ and its parameters)."
        )


OmegaConf.register_new_resolver(
    "resolve_hf_generation_module_on_transformer_model",
    resolve_hf_generation_module_on_transformer_model,
)
