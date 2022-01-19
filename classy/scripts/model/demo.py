import argparse
import itertools
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import streamlit as st
import torch
from omegaconf import OmegaConf

from classy.data.data_drivers import ClassySample, get_data_driver
from classy.utils.lightning import (
    load_classy_module_from_checkpoint,
    load_prediction_dataset_conf_from_checkpoint,
    load_training_conf_from_checkpoint,
)


def auto_infer_examples(
    task: str, model_checkpoint_path: str
) -> Tuple[str, List[ClassySample]]:
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


def tabbed_navigation(
    tabs: Dict[str, Tuple[str, Callable[[], None]]], default_tab: Optional[str] = None
):
    # adapted from https://discuss.streamlit.io/t/multiple-tabs-in-streamlit/1100/7
    # tabs is a dictionary that goes from key to (tab title, tab function)
    # e.g.: home: (Home, render_home)
    # default_tab points at a tab key (so 'home', not 'Home')

    st.markdown(
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" '
        'integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">',
        unsafe_allow_html=True,
    )

    query_params = st.experimental_get_query_params()
    tab_keys = list(tabs)
    default_tab = default_tab or tab_keys[0]
    if "tab" in query_params:
        active_tab = query_params["tab"][0]
    else:
        active_tab = default_tab

    if active_tab not in tab_keys:
        st.experimental_set_query_params(tab=default_tab)
        active_tab = default_tab

    li_items = "".join(
        f"""
        <li class="nav-item">
            <a class="nav-link{' active' if t == active_tab else ''}" href="/?tab={t}">{tabs.get(t)[0]}</a>
        </li>
        """
        for t in tab_keys
    )
    tabs_html = f"""
        <ul class="nav nav-tabs">
        {li_items}
        </ul>
    """

    st.markdown(tabs_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if active_tab not in tabs:
        st.error("Something has gone terribly wrong.")
        st.stop()

    tabs.get(active_tab)[1]()


def demo(
    model_checkpoint_path: str,
    cuda_device: int,
    prediction_params: Optional[str] = None,
):
    st.set_page_config(
        page_title="Classy demo",
        layout="wide",
        page_icon="https://sunglasses-ai.github.io/classy/img/CLASSY.svg",
    )

    @st.cache(allow_output_mutation=True)
    def load_resources():
        config = load_training_conf_from_checkpoint(
            model_checkpoint_path, post_trainer_init=False
        )
        model = load_classy_module_from_checkpoint(model_checkpoint_path)
        model.to(torch.device(cuda_device if cuda_device != -1 else "cpu"))
        model.freeze()

        if prediction_params is not None:
            model.load_prediction_params(dict(OmegaConf.load(prediction_params)))

        dataset_conf = load_prediction_dataset_conf_from_checkpoint(
            model_checkpoint_path
        )
        inference_message, inferred_examples = auto_infer_examples(
            model.task, model_checkpoint_path
        )

        # mock call to load resources
        next(
            model.predict(samples=[inferred_examples[0]], dataset_conf=dataset_conf),
            None,
        )

        return config, model, dataset_conf, (inference_message, inferred_examples)

    (
        config,
        model,
        dataset_conf,
        (inference_message, inferred_examples),
    ) = load_resources()

    # todo make this a param, allows to reload prediction params at each query (useful for model inspection)
    if prediction_params is not None:
        model.load_prediction_params(dict(OmegaConf.load(prediction_params)))

    # plot side bar
    st.sidebar.write(
        "<img src='https://sunglasses-ai.github.io/classy/img/CLASSY.svg' width='100%' height='100%' />",
        unsafe_allow_html=True,
    )

    st.sidebar.title("Classy Demo")
    with st.sidebar.expander(label="Task", expanded=True):
        model.ui_render_task_in_sidebar()

    with st.sidebar.expander(label="Model Info"):
        st.markdown(
            f"""
            * **model**: {model_checkpoint_path}
            * **device**: `{model.device}`
        """
        )

    def render_config():
        from classy.utils.rich_config import get_rich_tree_config, rich_to_html

        cfg_tree = get_rich_tree_config(config)
        st.write(rich_to_html(cfg_tree), unsafe_allow_html=True)

    def render_model_demo():
        # read input
        sample = model.ui_read_input(
            inference_message=inference_message, inferred_examples=inferred_examples
        )

        if sample is not None:
            # predict
            start = time.perf_counter()
            sample = next(model.predict(samples=[sample], dataset_conf=dataset_conf))
            end = time.perf_counter()

            # render output
            model.ui_render(sample, time=end - start)

    tabs = dict(
        model=("Model demo", render_model_demo), config=("Config", render_config)
    )

    tabbed_navigation(tabs, "model")


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
    parser.add_argument(
        "model_checkpoint", type=str, help="Path to pl_modules checkpoint"
    )
    parser.add_argument(
        "named_params_workaround",
        default=[],
        nargs="*",
        help="Named argument without --. Ex: prediction_params <path> cuda_device 0",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
