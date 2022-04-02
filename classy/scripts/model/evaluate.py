import itertools
import logging
from typing import Callable, Dict, List, Optional, Union

import hydra
import torch
from omegaconf import OmegaConf

from classy.data.data_drivers import ClassySample, DataDriver, get_data_driver
from classy.utils.lightning import (
    load_classy_module_from_checkpoint,
    load_prediction_dataset_conf_from_checkpoint,
    load_training_conf_from_checkpoint,
)


def evaluate(
    model_checkpoint_path: str,
    cuda_device: int,
    token_batch_size: int,
    input_path: Union[str, Dict[str, DataDriver]],
    output_type: Optional[str] = None,
    output_path: Optional[str] = None,
    evaluate_config_path: Optional[str] = None,
    prediction_params: Optional[str] = None,
    metrics_fn: Optional[
        Callable[
            [str, List[ClassySample]],
            Dict,
        ]
    ] = None,
):

    # load model
    model = load_classy_module_from_checkpoint(model_checkpoint_path)
    model.to(torch.device(cuda_device if cuda_device != -1 else "cpu"))
    model.freeze()

    if prediction_params is not None:
        model.load_prediction_params(dict(OmegaConf.load(prediction_params)))

    # load dataset conf and driver
    dataset_conf = load_prediction_dataset_conf_from_checkpoint(model_checkpoint_path)

    if isinstance(input_path, str):
        input_extension = input_path.split(".")[-1]
        data_driver = get_data_driver(model.task, input_extension)
        dataset_bundle = {input_path: data_driver}
    elif isinstance(input_path, dict):
        dataset_bundle = input_path
    else:
        logging.error("input_path must be a str or a DictConfig")
        raise ValueError

    # load evaluation metric
    if metrics_fn is not None:
        assert (
            evaluate_config_path is None
        ), "At most one between metrics_fn and evaluate_config_path can be provided"
    elif evaluate_config_path is not None:
        metrics_fn = hydra.utils.instantiate(OmegaConf.load(evaluate_config_path))
    else:
        evaluation_conf = load_training_conf_from_checkpoint(
            model_checkpoint_path
        ).evaluation
        metrics_fn = hydra.utils.instantiate(evaluation_conf)

    # predict
    predicted_samples = list(
        model.predict(
            model=model,
            samples=itertools.chain(
                *[dd.read_from_path(p) for p, dd in dataset_bundle.items()]
            ),
            dataset_conf=dataset_conf,
            token_batch_size=token_batch_size,
            progress_bar=True,
        )
    )

    # dump predictions if requested
    if output_path is not None:
        output_data_driver = get_data_driver(model.task, output_path.split(".")[-1])
        output_data_driver.save(
            predicted_samples, output_path, use_predicted_annotation=True
        )

    # run evaluation and print metrics
    result = metrics_fn(input_path, predicted_samples)

    def to_primitives(dictionary):
        def to_primitive(v):
            if hasattr(v, "item"):
                return v.item()
            return v

        return {k: to_primitive(v) for k, v in dictionary.items()}

    result = to_primitives(result)

    output_type = output_type or "tree"

    if output_type == "json":
        import json

        print(json.dumps(result))

    elif output_type == "list":
        for metric_name, metric_f in result.items():
            print(f"* {metric_name}: {metric_f}")

    elif output_type == "tree":
        from classy.utils.rich_config import print_config

        # taken from https://stackoverflow.com/a/35508197/1908499
        def nest_dict(flat_dict, sep="_"):
            """Return nested dict by splitting the keys on a delimiter."""
            tree = {}
            for key, val in flat_dict.items():
                t = tree
                prev = None
                for part in key.split(sep):
                    if prev is not None:
                        t = t.setdefault(prev, {})
                    prev = part
                else:
                    t.setdefault(prev, val)
            return tree

        c = OmegaConf.create(nest_dict(result))
        print_config(c, tree_label="<scores>")
