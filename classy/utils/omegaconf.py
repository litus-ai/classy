import copy
import re

from omegaconf import DictConfig, OmegaConf


def adapt_dataset_from(training_dataset: DictConfig, setting: str):
    if setting == "validation":
        validation_dataset = copy.deepcopy(training_dataset)
        validation_dataset["materialize"] = True
        return validation_dataset
    elif setting == "prediction":
        prediction_dataset = copy.deepcopy(training_dataset)
        prediction_dataset["_target_"] = re.sub(
            ".from_file$", ".from_samples", prediction_dataset["_target_"]
        )
        prediction_dataset["min_length"] = -1
        prediction_dataset["max_length"] = -1
        prediction_dataset["for_inference"] = True
        return prediction_dataset
    else:
        raise ValueError(
            f"Setting {setting} not supported. Choose between [validation, prediction] or change config."
        )


OmegaConf.register_new_resolver("adapt_dataset_from", adapt_dataset_from)
