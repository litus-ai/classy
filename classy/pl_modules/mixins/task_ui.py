import collections
import itertools
import re
from typing import Union, List, Optional, Tuple

import streamlit as st
from annotated_text import annotated_text

from classy.data.data_drivers import (
    SentencePairSample,
    SequenceSample,
    TokensSample,
    get_data_driver,
)
from classy.scripts.cli.evaluate import automatically_infer_test_path
from classy.utils.streamlit import get_md_200_random_color_generator


class TaskUIMixin:
    def render_task_in_sidebar(self):
        raise NotImplementedError

    def read_input(
        self, inference_message: str, inferred_examples: List[Union[SentencePairSample, SequenceSample, TokensSample]]
    ) -> Optional[Union[SentencePairSample, SequenceSample, TokensSample]]:
        raise NotImplementedError

    def render(self, predicted_sample: Union[SentencePairSample, SequenceSample, TokensSample]):
        raise NotImplementedError


class SentencePairTaskUIMixin(TaskUIMixin):

    truncate_k = 40

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

    def read_input(
        self, inference_message: str, inferred_examples: List[SentencePairSample]
    ) -> Optional[SentencePairSample]:
        # tuple can't be used for selection boxes, let's use incipts
        option2example = {}
        for ie in inferred_examples:
            option2example[f"({ie.sentence1[: self.truncate_k]}, {ie.sentence2[: self.truncate_k]})"] = (
                ie.sentence1,
                ie.sentence2,
            )
        # build selection box
        selection_message = (
            inference_message.rstrip(".")
            + f". Examples are in the format (<sentence1>, <sentence2>); for space constraints, only the first {self.truncate_k} characters of each sentence are shown."
        )
        selected_option = st.selectbox(selection_message, options=list(option2example.keys()), index=0)
        sentence1 = st.text_area("First input sequence", option2example[selected_option][0])
        sentence2 = st.text_area("Second input sequence", option2example[selected_option][1])
        if st.button("Classify", key="classify"):
            return SentencePairSample(sentence1=sentence1, sentence2=sentence2)
        return None

    def render(self, predicted_sample: SentencePairSample):
        st.success(f"Model classified input with label: {predicted_sample.label}")


class SequenceTaskUIMixin(TaskUIMixin):
    def render_task_in_sidebar(self):
        st.sidebar.header("Task")
        st.sidebar.markdown(
            f"""
                * **task**: Sequence Classification
                * **input**: String sequence
            """
        )

    def read_input(self, inference_message: str, inferred_examples: List[SequenceSample]) -> Optional[SequenceSample]:
        placeholder = st.selectbox(inference_message, options=[ie.sequence for ie in inferred_examples], index=0)
        input_text = st.text_area("Input sequence to classify", placeholder)
        if st.button("Classify", key="classify"):
            return SequenceSample(sequence=input_text)
        return None

    def render(self, predicted_sample: SequenceSample):
        st.success(f"Model classified input with label: {predicted_sample.label}")


class TokenTaskUIMixin(TaskUIMixin):

    color_generator = get_md_200_random_color_generator()
    color_mapping = collections.defaultdict(lambda: TokenTaskUIMixin.color_generator())

    def render_task_in_sidebar(self):
        st.sidebar.header("Task")
        st.sidebar.markdown(
            f"""
            * **task**: Token Classification
            * **input**: Space-separeted list of tokens
        """
        )

    def read_input(self, inference_message: str, inferred_examples: List[TokensSample]) -> Optional[TokensSample]:
        placeholder = st.selectbox(
            inference_message, options=[" ".join(ie.tokens) for ie in inferred_examples], index=0
        )
        input_text = st.text_area("Space-separeted list of tokens to classify", placeholder)
        if st.button("Classify", key="classify"):
            return TokensSample(tokens=input_text.split(" "))
        return None

    def render(self, predicted_sample: TokensSample):

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

        annotated_text(*[(f" {t} " if l is None else (t, l, self.color_mapping[l])) for t, l in zip(tokens, labels)])
