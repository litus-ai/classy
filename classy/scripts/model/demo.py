import argparse
import itertools
from typing import List, Union, Tuple

import streamlit as st
import torch

from classy.data.data_drivers import SentencePairSample, SequenceSample, TokensSample, get_data_driver
from classy.scripts.cli.evaluate import automatically_infer_test_path
from classy.utils.lightning import load_classy_module_from_checkpoint, load_prediction_dataset_conf_from_checkpoint


def auto_infer_examples(
    task: str, model_checkpoint_path: str
) -> Tuple[str, List[Union[SentencePairSample, SequenceSample, TokensSample]]]:

    try:
        test_path = automatically_infer_test_path(model_checkpoint_path)
        return "Examples from test set", list(
            itertools.islice(get_data_driver(task, test_path.split(".")[-1]).read_from_path(test_path), 5)
        )
    except ValueError:
        pass

    # todo try safe dumped option

    raise ValueError("Could not find any examples")


def demo(model_checkpoint_path: str, cuda_device: int):
    @st.cache(allow_output_mutation=True)
    def load_resources():
        model = load_classy_module_from_checkpoint(model_checkpoint_path)
        model.to(torch.device(cuda_device if cuda_device != -1 else "cpu"))
        model.freeze()

        dataset_conf = load_prediction_dataset_conf_from_checkpoint(model_checkpoint_path)
        inference_message, inferred_examples = auto_infer_examples(model.task, model_checkpoint_path)

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
        _, prediction = next(model.predict(samples=[sample], dataset_conf=dataset_conf))
        sample.update_classification(prediction)

        # render output
        model.render(sample)


def main():
    args = parse_args()
    demo(model_checkpoint_path=args.model_checkpoint, cuda_device=args.cuda_device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_checkpoint", type=str, help="Path to pl_modules checkpoint")
    parser.add_argument("cuda_device", type=int, default=-1, nargs="?", help="Cuda device")
    return parser.parse_args()


if __name__ == "__main__":
    main()
