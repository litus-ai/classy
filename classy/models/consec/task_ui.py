import collections
import re
from typing import List, Optional, Tuple

import streamlit as st
from annotated_text import annotated_text

from classy.data.data_drivers import TokensSample
from classy.pl_modules.mixins.task_ui import TokenTaskUIMixin


class ConSeCTaskUIMixin(TokenTaskUIMixin):
    # todo we are not handling multiwords

    def render_task_in_sidebar(self):
        st.sidebar.header("Task")
        st.sidebar.markdown(
            f"""
            * **task**: Token Classification
            * **input**: Space-separeted list of tokens, pos, lemmas and target to disambiguate
        """
        )

    def read_input(self, inference_message: str, inferred_examples: List[TokensSample]) -> Optional[TokensSample]:
        option2inferred_example = {}
        for ie in inferred_examples:
            option2inferred_example[" ".join(ie.tokens)] = ie
        option = st.selectbox(inference_message, options=list(option2inferred_example.keys()), index=0)
        text = st.text_area(
            "Space-separeted list of tokens to classify", " ".join(option2inferred_example[option].tokens)
        )
        pos = st.text_area(
            "Space-separeted list of POS, corresponding to their token counterparts",
            " ".join(option2inferred_example[option].pos),
        )
        lemma = st.text_area(
            "Space-separeted list of POS, corresponding to their token counterparts",
            " ".join(option2inferred_example[option].lemma),
        )
        target = st.text_area(
            "Space-separeted list of positions, corresponding to the tokens you want to disambiguate",
            " ".join(map(str, option2inferred_example[option].target)),
        )
        if st.button("Classify", key="classify"):
            return TokensSample(
                document_id="d0",
                sentence_id="d0.s0",
                tokens=text.split(" "),
                pos=pos.split(" "),
                lemma=lemma.split(" "),
                target=[int(_t) for _t in target.split(" ")],
                instance_ids=target.split(" "),
            )
        return None

    def render(self, predicted_sample: TokensSample):
        tokens = predicted_sample.tokens
        labels = [None] * len(tokens)
        for t, l in zip(predicted_sample.target, predicted_sample.labels):
            labels[t] = l
        annotated_text(*[(f" {t} " if l is None else (t, l, self.color_mapping[l])) for t, l in zip(tokens, labels)])
