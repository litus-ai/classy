import argparse
import itertools
import time
from pathlib import Path
from typing import List, Union, Tuple, Dict, Optional

import streamlit as st
import torch
from omegaconf import OmegaConf

from classy.data.data_drivers import SentencePairSample, SequenceSample, TokensSample, get_data_driver
from classy.scripts.cli.evaluate import automatically_infer_test_path
from classy.utils.lightning import load_classy_module_from_checkpoint, load_prediction_dataset_conf_from_checkpoint


def auto_infer_examples(
    task: str, model_checkpoint_path: str
) -> Tuple[str, List[Union[SentencePairSample, SequenceSample, TokensSample]]]:
    experiment_folder = Path(model_checkpoint_path).parent.parent
    if (experiment_folder / "data" / "examples-test.jsonl").exists():
        return "Examples from test", list(
            itertools.islice(
                get_data_driver(task, "jsonl").read_from_path(
                    str((experiment_folder / "data" / "examples-test.jsonl"))
                ),
                5,
            )
        )
    else:
        assert (experiment_folder / "data" / "examples-validation.jsonl").exists()
        return "Examples from validation", list(
            itertools.islice(
                get_data_driver(task, "jsonl").read_from_path(
                    str((experiment_folder / "data" / "examples-validation.jsonl"))
                ),
                5,
            )
        )


def demo(model_checkpoint_path: str, cuda_device: int, prediction_params: Optional[str] = None):
    @st.cache(allow_output_mutation=True)
    def load_resources():
        model = load_classy_module_from_checkpoint(model_checkpoint_path)
        model.to(torch.device(cuda_device if cuda_device != -1 else "cpu"))
        model.freeze()

        if prediction_params is not None:
            model.load_prediction_params(dict(OmegaConf.load(prediction_params)))

        dataset_conf = load_prediction_dataset_conf_from_checkpoint(model_checkpoint_path)
        inference_message, inferred_examples = auto_infer_examples(model.task, model_checkpoint_path)

        # mock call to load resources
        next(model.predict(samples=[inferred_examples[0]], dataset_conf=dataset_conf), None)

        return model, dataset_conf, (inference_message, inferred_examples)

    model, dataset_conf, (inference_message, inferred_examples) = load_resources()

    # plot side bar
    st.sidebar.title("Sunglasses-AI 🕶️")
    st.sidebar.title("Classy Demo")
    model.render_task_in_sidebar()
    st.sidebar.header("Model Info")
    st.sidebar.markdown(
        f"""
        * **model**: {model_checkpoint_path}
        * **device**: {"gpu" if cuda_device >= 0 else "cpu"}
    """
    )

    # read input
    sample = model.read_input(inference_message=inference_message, inferred_examples=inferred_examples)

    if sample is not None:

        # predict
        start = time.perf_counter()
        _, prediction = next(model.predict(samples=[sample], dataset_conf=dataset_conf))
        end = time.perf_counter()
        sample.update_classification(prediction)

        # render output
        model.render(sample, time=end - start)


def main():
    args = parse_args()
    # todo proper solution with argparse named arguments
    named_arguments = {"cuda_device": -1}
    named_params_workaround = args.named_params_workaround
    if len(named_params_workaround) > 0:
        for i in range(0, len(named_params_workaround), 2):
            k, v = named_params_workaround[i], named_params_workaround[i + 1]
            if k == "cuda_device":
                v = int(v)
            named_arguments[k] = v
    # actual main
    demo(model_checkpoint_path=args.model_checkpoint, **named_arguments)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_checkpoint", type=str, help="Path to pl_modules checkpoint")
    parser.add_argument(
        "named_params_workaround",
        default=[],
        nargs="*",
        help="Named argument without --. Ex: prediction_params <path> cuda_device 0",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
