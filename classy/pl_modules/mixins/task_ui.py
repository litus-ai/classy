import collections
import html
import json
import re
from typing import List, Optional

import streamlit as st
from annotated_text import annotation

from classy.data.data_drivers import (
    GENERATION,
    JSONL,
    QA,
    SENTENCE_PAIR,
    SEQUENCE,
    TOKEN,
    ClassySample,
    GenerationSample,
    QASample,
    SentencePairSample,
    SequenceSample,
    TokensSample,
    get_data_driver,
)
from classy.utils.streamlit import get_md_200_random_color_generator


class TaskUIMixin:
    def ui_render_task_in_sidebar(self):
        raise NotImplementedError

    def ui_read_input(
        self,
        inference_message: str,
        inferred_examples: List[ClassySample],
    ) -> Optional[ClassySample]:
        raise NotImplementedError

    def ui_render(
        self,
        predicted_sample: ClassySample,
        time: float,
    ):
        raise NotImplementedError


class SentencePairTaskUIMixin(TaskUIMixin):

    __data_driver = get_data_driver(SENTENCE_PAIR, JSONL)
    truncate_k = 40

    def ui_render_task_in_sidebar(self):
        st.markdown(
            f"""
                * **task**: Sentence-Pair Classification
                * **input**: Pair of input sentences
            """
        )

    def ui_read_input(
        self, inference_message: str, inferred_examples: List[SentencePairSample]
    ) -> Optional[SentencePairSample]:
        # tuple can't be used for selection boxes, let's use incipts
        option2example = {}
        for ie in inferred_examples:
            option2example[
                f"({ie.sentence1[: self.truncate_k]}, {ie.sentence2[: self.truncate_k]})"
            ] = (
                ie.sentence1,
                ie.sentence2,
            )
        # build selection box
        selection_message = (
            inference_message.rstrip(".")
            + f". Examples are in the format (<sentence1>, <sentence2>); for space constraints, only the first {self.truncate_k} characters of each sentence are shown."
        )
        selected_option = st.selectbox(
            selection_message, options=list(option2example.keys()), index=0
        )
        sentence1 = st.text_area(
            "First input sequence", option2example[selected_option][0]
        )
        sentence2 = st.text_area(
            "Second input sequence", option2example[selected_option][1]
        )
        if st.button("Classify", key="classify"):
            sample = json.dumps({"sentence1": sentence1, "sentence2": sentence2})
            return next(self.__data_driver.read([sample]))
        return None

    def ui_render(self, predicted_sample: SentencePairSample, time: float):
        st.markdown(
            f"""
            <div>
                <div class="stAlert">
                    <p>Model classified input with label: <b>{predicted_sample.predicted_annotation}</b></p>
                </div>
                <p></p>
                <div style="text-align: right"><p style="color: gray">Time: {time:.2f}s</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


class SequenceTaskUIMixin(TaskUIMixin):
    __data_driver = get_data_driver(SEQUENCE, JSONL)

    def ui_render_task_in_sidebar(self):
        st.markdown(
            f"""
                * **task**: Sequence Classification
                * **input**: String sequence
            """
        )

    def ui_read_input(
        self, inference_message: str, inferred_examples: List[SequenceSample]
    ) -> Optional[SequenceSample]:
        placeholder = st.selectbox(
            inference_message,
            options=[ie.sequence for ie in inferred_examples],
            index=0,
        )
        input_text = st.text_area("Input sequence to classify", placeholder)
        if st.button("Classify", key="classify"):
            sample = json.dumps({"sequence": input_text})
            return next(self.__data_driver.read([sample]))
        return None

    def ui_render(self, predicted_sample: SequenceSample, time: float):
        st.markdown(
            f"""
            <div>
                <div class="stAlert">
                    <p>Model classified input with label: <b>{predicted_sample.predicted_annotation}</b></p>
                </div>
                <p></p>
                <div style="text-align: right"><p style="color: gray">Time: {time:.2f}s</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


class TokenTaskUIMixin(TaskUIMixin):
    __data_driver = get_data_driver(TOKEN, JSONL)
    color_generator = get_md_200_random_color_generator()
    color_mapping = collections.defaultdict(lambda: TokenTaskUIMixin.color_generator())

    def ui_render_task_in_sidebar(self):
        st.markdown(
            f"""
                * **task**: Token Classification
                * **input**: Space-separeted list of tokens
            """
        )

    def ui_read_input(
        self, inference_message: str, inferred_examples: List[TokensSample]
    ) -> Optional[TokensSample]:
        placeholder = st.selectbox(
            inference_message,
            options=[" ".join(ie.tokens) for ie in inferred_examples],
            index=0,
        )
        input_text = st.text_area(
            "Space-separeted list of tokens to classify", placeholder
        )
        if st.button("Classify", key="classify"):
            sample = json.dumps({"tokens": input_text.split(" ")})
            return next(self.__data_driver.read([sample]))
        return None

    def ui_render(self, predicted_sample: TokensSample, time: float):

        tokens, labels = predicted_sample.tokens, predicted_sample.predicted_annotation

        # check if any token encodings (e.g. bio) are used
        if all(
            l == "O" or re.fullmatch("^[BI]-.*$", l)
            for l in predicted_sample.predicted_annotation
        ):
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
                annotated_html_components.append(
                    str(annotation(*(t, l, self.color_mapping[l])))
                )

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


class QATaskUIMixin(TaskUIMixin):
    __data_driver = get_data_driver(QA, JSONL)
    truncate_k = 40

    def ui_render_task_in_sidebar(self):
        st.markdown(
            f"""
                * **task**: QA
                * **input**: context and question
            """
        )

    def ui_read_input(
        self, inference_message: str, inferred_examples: List[QASample]
    ) -> Optional[QASample]:
        # tuple can't be used for selection boxes, let's use incipts
        option2example = {}
        for ie in inferred_examples:
            option2example[
                f"({ie.question[: self.truncate_k]}, {ie.context[: self.truncate_k]})"
            ] = (
                ie.question,
                ie.context,
            )
        # build selection box
        selection_message = (
            inference_message.rstrip(".")
            + f". Examples are in the format (<question>, <context>); for space constraints, only the first {self.truncate_k} characters of both are shown."
        )
        selected_option = st.selectbox(
            selection_message, options=list(option2example.keys()), index=0
        )
        # build input area
        question = st.text_area("Question", option2example[selected_option][0])
        context = st.text_area("Context", option2example[selected_option][1])
        if st.button("Classify", key="classify"):
            sample = json.dumps({"question": question, "context": context})
            return next(self.__data_driver.read([sample]))
        return None

    def ui_render(self, predicted_sample: QASample, time: float):
        char_start, char_end = predicted_sample.predicted_annotation
        annotated_html_components = [
            str(html.escape(f"{predicted_sample.context[: char_start]} ")),
            str(
                annotation(
                    f"{predicted_sample.context[char_start: char_end]}",
                    background="#f1e740",
                )
            ),
            str(html.escape(f"{predicted_sample.context[char_end:]} ")),
        ]
        print(annotated_html_components)
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


class GenerationTaskUIMixin(TaskUIMixin):
    __data_driver = get_data_driver(GENERATION, JSONL)

    def ui_render_task_in_sidebar(self):
        st.markdown(
            f"""
                * **task**: QA
                * **input**: source sequence and, optionally (depending on the model), source and target language
            """
        )

    def ui_read_input(
        self,
        inference_message: str,
        inferred_examples: List[GenerationSample],
    ) -> Optional[GenerationSample]:
        # tuple can't be used for selection boxes, let's use incipts
        option2example = {}
        for ie in inferred_examples:
            option2example[f"{ie.source_sequence}"] = (
                ie.source_sequence,
                ie.source_language if ie.source_language is not None else "",
                ie.target_language if ie.target_language is not None else "",
            )
        # actual reading
        selected_option = st.selectbox(
            inference_message, options=list(option2example.keys()), index=0
        )
        source_sequence = st.text_area(
            "Source sequence", option2example[selected_option][0]
        )
        source_language = st.text_input(
            "Source language (empty to set it to None)",
            option2example[selected_option][1],
        ).strip()
        target_language = st.text_input(
            "Target language (empty to set it to None)",
            option2example[selected_option][2],
        ).strip()
        if st.button("Classify", key="classify"):
            sample = json.dumps(
                dict(
                    source_sequence=source_sequence,
                    source_language=source_language or None,
                    target_language=target_language or None,
                )
            )
            return next(self.__data_driver.read([sample]))
        return None

    def ui_render(self, predicted_sample: GenerationSample, time: float):
        st.markdown(
            f"""
                    <div>
                        <div class="stAlert">
                            <p>Model generated: <br>{predicted_sample.predicted_annotation}</p>
                        </div>
                        <p></p>
                        <div style="text-align: right"><p style="color: gray">Time: {time:.2f}s</p></div>
                    </div>
                    """,
            unsafe_allow_html=True,
        )
