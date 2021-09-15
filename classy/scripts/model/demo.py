import argparse
import collections
import html
import itertools
import random
import re
import time
from typing import Union, Callable, List, Optional, Tuple

import streamlit as st
import torch
from annotated_text import annotation

from classy.data.data_drivers import (
    SentencePairSample,
    SequenceSample,
    TokensSample,
    TOKEN,
    get_data_driver,
    SENTENCE_PAIR,
    SEQUENCE,
)
from classy.scripts.cli.evaluate import automatically_infer_test_path
from classy.scripts.model.predict import predict
from classy.utils.lightning import load_classy_module_from_checkpoint, load_prediction_dataset_conf_from_checkpoint


def get_random_color_generator() -> Callable[[], str]:

    # colors taken from https://gist.githubusercontent.com/daniellevass/b0b8cfa773488e138037/raw/d2182c212a4132c0f3bb093fd0010395f927a219/android_material_design_colours.xml
    # md_.*_200
    default_colors = [
        "#81D4fA",
        "#B39DDB",
        "#80CBC4",
        "#C5E1A5",
        "#EF9A9A",
        "#F48FB1",
        "#9FA8DA",
        "#CE93D8",
        "#FFCC80",
        "#FFE082",
        "#80DEEA",
        "#A5D6A7",
        "#FFAB91",
        "#90CAF9",
        "#EEEEEE",
        "#BCAAA4",
        "#E6EE9C",
        "#FFF590",
        "#B0BBC5",
    ]

    colors = iter(default_colors)

    def f():
        try:
            return next(colors)
        except StopIteration:
            return "#%06x" % random.randint(0x000000, 0xFFFFFF)

    return f


class TaskUI:
    @staticmethod
    def from_task(task: str, model_checkpoint_path: str):
        if task == SENTENCE_PAIR:
            return SentencePairTaskUI(task, model_checkpoint_path)
        elif task == SEQUENCE:
            return SequenceTaskUI(task, model_checkpoint_path)
        elif task == TOKEN:
            return TokenTaskUI(task, model_checkpoint_path)
        else:
            raise ValueError

    def __init__(self, task: str, model_checkpoint_path: str):
        self.task = task
        self.model_checkpoint_path = model_checkpoint_path
        self.__cached_examples, self.__infer_failed = None, False

    def auto_infer_examples(self) -> List[Union[SentencePairSample, SequenceSample, TokensSample]]:
        if not self.__infer_failed and self.__cached_examples is None:
            try:
                test_path = automatically_infer_test_path(self.model_checkpoint_path)
                self.__cached_examples = list(
                    itertools.islice(get_data_driver(self.task, test_path.split(".")[-1]).read_from_path(test_path), 5)
                )
            except ValueError:
                self.__infer_failed = True
        return self.__cached_examples

    def render_task_in_sidebar(self):
        raise NotImplementedError

    def read_input(self) -> Union[SentencePairSample, SequenceSample, TokensSample]:
        raise NotImplementedError

    def render(self, predicted_sample: Union[SentencePairSample, SequenceSample, TokensSample], time: float):
        raise NotImplementedError


class SentencePairTaskUI(TaskUI):
    def __init__(self, task: str, model_checkpoint_path: str):
        super().__init__(task, model_checkpoint_path)
        self.truncate_k = 40

    def render_task_in_sidebar(self):
        st.sidebar.header("Task")
        st.sidebar.markdown(
            f"""
                * **task**: Sentence-Pair Classification
                * **input**: Pair of input sentences
            """
        )

    def get_examples(self) -> Tuple[List[Tuple[str, str]], bool]:
        inferred_examples = self.auto_infer_examples()
        if inferred_examples is not None:
            return [(ie.sentence1, ie.sentence2) for ie in inferred_examples], True
        else:
            # todo these examples are not ideal for pretty much any sentence-pair task, suggestions?
            return [
                ("Classy is really nice for token classification!", "Classy is a product of SunglassesAI."),
                (
                    "It focuses on classification tasks at token- and sequence level.",
                    "Ease of usability is our primary objective.",
                ),
            ], False

    def read_input(self) -> Optional[SentencePairSample]:
        examples, auto_infer = self.get_examples()
        # tuple can't be used for selection boxes, let's use incipts
        option2example = {}
        for sentence1, sentence2 in examples:
            option2example[f"({sentence1[: self.truncate_k]}, {sentence2[: self.truncate_k]})"] = (sentence1, sentence2)
        # build selection box
        selection_message = "Examples from test set." if auto_infer else "Examples."
        selection_message += f" Examples are in the format (<sentence1>, <sentence2>); for space constraints, only the first {self.truncate_k} characters of each sentence are shown."
        selected_option = st.selectbox(selection_message, options=list(option2example.keys()), index=0)
        sentence1 = st.text_area("First input sequence", option2example[selected_option][0])
        sentence2 = st.text_area("Second input sequence", option2example[selected_option][1])
        if st.button("Classify", key="classify"):
            return SentencePairSample(sentence1=sentence1, sentence2=sentence2)
        return None

    def render(self, predicted_sample: SentencePairSample, time: float):
        st.markdown(
            f"""
            <div>
                <div class="stAlert">
                    <p>Model classified input with label: <b>{predicted_sample.label}</b></p>
                </div>
                <p></p>
                <div style="text-align: right"><p style="color: gray">Time: {time:.2f}s</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


class SequenceTaskUI(TaskUI):
    def render_task_in_sidebar(self):
        st.sidebar.header("Task")
        st.sidebar.markdown(
            f"""
                * **task**: Sequence Classification
                * **input**: String sequence
            """
        )

    def get_examples(self) -> Tuple[List[str], bool]:
        inferred_examples = self.auto_infer_examples()
        if inferred_examples is not None:
            return [ie.sequence for ie in inferred_examples], True
        else:
            return [
                "Classy is really nice for token classification!",
                "Classy is a product of SunglassesAI.",
                "It focuses on classification tasks at token- and sequence level.",
                "Ease of usability is our primary objective.",
                "If you are interested in generation instead, check out Classy's sibling, Genie!",
            ], False

    def read_input(self) -> Optional[SequenceSample]:
        examples, auto_infer = self.get_examples()
        placeholder = st.selectbox("Examples from test set" if auto_infer else "Examples", options=examples, index=0)
        input_text = st.text_area("Input sequence to classify", placeholder)
        if st.button("Classify", key="classify"):
            return SequenceSample(sequence=input_text)
        return None

    def render(self, predicted_sample: SequenceSample, time: float):
        st.markdown(
            f"""
            <div>
                <div class="stAlert">
                    <p>Model classified input with label: <b>{predicted_sample.label}</b></p>
                </div>
                <p></p>
                <div style="text-align: right"><p style="color: gray">Time: {time:.2f}s</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


class TokenTaskUI(TaskUI):
    def __init__(self, task: str, model_checkpoint_path: str):
        super().__init__(task, model_checkpoint_path)
        self.color_generator = get_random_color_generator()
        self.color_mapping = collections.defaultdict(lambda: self.color_generator())

    def render_task_in_sidebar(self):
        st.sidebar.header("Task")
        st.sidebar.markdown(
            f"""
            * **task**: Token Classification
            * **input**: Space-separeted list of tokens
        """
        )

    def get_examples(self) -> Tuple[List[str], bool]:
        inferred_examples = self.auto_infer_examples()
        if inferred_examples is not None:
            return [" ".join(ie.tokens) for ie in inferred_examples], True
        else:
            return [
                "Classy is really nice for token classification !",
                "Classy is a product of SunglassesAI .",
                "It focuses on classification tasks at token- and sequence level .",
                "Ease of usability is our primary objective .",
                "If you are interested in generation instead , check out Classy 's sibling, Genie !",
            ], False

    def read_input(self) -> Optional[TokensSample]:
        examples, auto_infer = self.get_examples()
        placeholder = st.selectbox("Examples from test set" if auto_infer else "Examples", options=examples, index=0)
        input_text = st.text_area("Space-separeted list of tokens to classify", placeholder)
        if st.button("Classify", key="classify"):
            return TokensSample(tokens=input_text.split(" "))
        return None

    def render(self, predicted_sample: TokensSample, time: float):

        tokens, labels = predicted_sample.tokens, predicted_sample.labels

        # check if any token encodings (e.g. bio) are used
        if all(l == "O" or re.fullmatch("^[BI]-.*$", l) for l in predicted_sample.labels):
            _tokens, _labels = [], []
            for t, l in zip(tokens, labels):
                if l.startswith("I"):
                    _tokens[-1] += f" {t}"
                else:
                    _tokens.append(t)
                    _labels.append(None if l == "O" else l[2:])
            tokens, labels = _tokens, _labels

        assert len(tokens) == len(labels)

        annotated_html_components = []
        for t, l in zip(tokens, labels):
            if l is None:
                annotated_html_components.append(str(html.escape(f" {t} ")))
            else:
                annotated_html_components.append(str(annotation(*(t, l, self.color_mapping[l]))))

        st.markdown(
            "\n".join(
                [
                    "<div>",
                    *annotated_html_components,
                    "<p></p>"
                    f'<div style="text-align: right"><p style="color: gray">Time: {time:.2f}s</p></div>'
                    "</div>",
                ]
            ),
            unsafe_allow_html=True,
        )


def demo(model_checkpoint_path: str, cuda_device: int):
    @st.cache(allow_output_mutation=True)
    def load_resources():
        model = load_classy_module_from_checkpoint(model_checkpoint_path)
        model.to(torch.device(cuda_device if cuda_device != -1 else "cpu"))
        model.freeze()

        dataset_conf = load_prediction_dataset_conf_from_checkpoint(model_checkpoint_path)
        task_ui = TaskUI.from_task(model.task, model_checkpoint_path)

        return model, dataset_conf, task_ui

    model, dataset_conf, task_ui = load_resources()

    # plot side bar
    st.sidebar.title("Sunglasses-AI 🕶️")
    st.sidebar.title("Classy Demo")
    task_ui.render_task_in_sidebar()
    st.sidebar.header("Model Info")
    st.sidebar.markdown(
        f"""
        * **model**: {model_checkpoint_path}
        * **device**: {"gpu" if cuda_device >= 0 else "cpu"}
    """
    )

    # read input
    sample = task_ui.read_input()

    if sample is not None:

        # predict
        start = time.perf_counter()
        _, prediction = next(predict(model=model, samples=[sample], dataset_conf=dataset_conf))
        end = time.perf_counter()
        sample.update_classification(prediction)

        # render output
        task_ui.render(sample, time=end - start)


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
