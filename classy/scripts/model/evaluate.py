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
    for metric_name, metric_f in result.items():
        print(f"* {metric_name}: {metric_f}")
