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
