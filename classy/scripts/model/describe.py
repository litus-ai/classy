import argparse
import collections
from typing import Iterable, List, Union, Optional, Tuple, Dict

import streamlit as st
import plotly.express as px

import numpy as np

from sacremoses import MosesTokenizer

from classy.data.data_drivers import (
    SequenceSample,
    SentencePairSample,
    TokensSample,
    QASample,
    get_data_driver,
    SEQUENCE,
    SENTENCE_PAIR,
    TOKEN,
    QA,
)

# colors
from classy.utils.streamlit import get_md_400_random_color_generator
from classy.utils.plotly import boxplot

colors_iter = get_md_400_random_color_generator()


def get_random_color() -> str:
    return colors_iter()


class UIMetric:
    """
    Base class for metrics that knows how to update themselves with an iterable of dataset samples and how to update
    the streamlit page on the base of the computed metric.
    """

    def title(self) -> Optional[Union[str, List[str]]]:
        raise NotImplementedError

    def description(self) -> Optional[Union[str, List[str]]]:
        raise NotImplementedError

    def write_body(self) -> None:
        raise NotImplementedError

    def is_writable(self) -> bool:
        raise NotImplementedError

    def update_metric(
        self, dataset_sample: Union[SequenceSample, SentencePairSample, TokensSample, QASample, str]
    ) -> None:
        raise NotImplementedError

    def write_metric(self) -> None:
        st.header(self.title())
        st.markdown(self.description())
        self.write_body()
        st.markdown("---")  # metrics separator


class InputLenUIMetric(UIMetric):
    """
    Simple metric to compute Avg, Max and Min length on the passed sequences (e.g. QA contexts). The length can be
    computed both on the number of characters and on the number of tokens depending on the input dataset_sample type.
    """

    def __init__(self, title: str, description: str):
        self._title = title
        self._description = description
        self.tokenized_input = None
        self._input_lens = []

    def title(self) -> Optional[str]:
        return self._title

    def description(self) -> Optional[Union[str, List[str]]]:
        return self._description

    def is_writable(self) -> bool:
        return len(self._input_lens) > 0

    def update_metric(
        self,
        dataset_sample: Union[
            str, List[str], Tuple[List[str], Union[SequenceSample, SentencePairSample, TokensSample, QASample]]
        ],
    ) -> None:
        """
        Update the metrics lengths store
        Args:
            dataset_sample: can be
                - str: a text sequence
                - List[str]: list of tokens
                - Tuple[List[str], Union[SequenceSample, SentencePairSample, TokensSample, QASample]]: a tuple
                    containing a list of tokens along with the original_sample

        Returns:
            None
        """
        if type(dataset_sample) == tuple:
            dataset_sample, _ = dataset_sample  # we don't need here the original sample

        if self.tokenized_input is None:
            self.tokenized_input = type(dataset_sample) == list

        # add the input length being either the number of chars (str) or the number of tokens (List[str])
        self._input_lens.append(len(dataset_sample))

    def write_body(self) -> None:
        st.text("")
        c1, c2, c3, c4 = st.columns(4)
        unit = "tokens" if self.tokenized_input else "characters"
        c1.metric(label=f"Avg len in {unit}", value="{:.2f}".format(sum(self._input_lens) / len(self._input_lens)))
        c2.metric(label=f"Min len in {unit}", value="{:.2f}".format(min(self._input_lens)))
        c3.metric(label=f"Max len in {unit}", value="{:.2f}".format(max(self._input_lens)))
        c4.metric(label=f"Total number of samples", value=f"{len(self._input_lens)}")
        char_len_boxplot = boxplot(
            y=np.array(self._input_lens), x_name=self.title(), y_name=f"length in {unit}", color=get_random_color()
        )
        st.write(char_len_boxplot)


class ClassSpecificInputLenUIMetric(UIMetric):
    """
    The class specific variant of "InputLenUIMetric". It computes all the stats of the InputLenUIMetric but for each
    class separated.
    """

    def __init__(self, title_format: str, description_format: str):
        self.title_format = title_format
        self.description_format = description_format
        self.class2input_len_ui_metric: Dict[str, InputLenUIMetric] = dict()

    def title(self) -> Optional[Union[str, List[str]]]:
        return [um.title() for um in self.class2input_len_ui_metric.values()]

    def description(self) -> Optional[Union[str, List[str]]]:
        raise NotImplementedError

    def write_body(self) -> None:
        raise NotImplementedError

    def write_metric(self) -> None:
        for ui_metric in self.class2input_len_ui_metric.values():
            if not ui_metric.is_writable():
                continue
            ui_metric.write_metric()

    def is_writable(self) -> bool:
        return len(self.class2input_len_ui_metric) > 0

    def update_metric(self, dataset_sample) -> None:
        sequence, dataset_sample = dataset_sample

        label = dataset_sample.get_current_classification()

        if label is None:
            return

        if type(label) != str:  # TODO: is it correct? does the string means that we have just one class
            print("Labels at this moment can only be classes.")
            raise NotImplementedError

        if label not in self.class2input_len_ui_metric:
            self.class2input_len_ui_metric[label] = InputLenUIMetric(
                self.title_format.format(label), self.description_format.format(label)
            )

        self.class2input_len_ui_metric[label].update_metric(sequence)


class LambdaWrapperUIMetric(UIMetric):
    """
    UIMetric wrapper that let you define a lambda function as an indirection between the dataset_sample and the
    update_metric. e.g. "lambda dataset_sample: tokenize(dataset_sample.sequence)" in order to pass the input already
    tokenized.
    """

    def __init__(
        self,
        ui_metric: Union[UIMetric, List[UIMetric]],
        dataset_sample_modifier,
    ):
        self.ui_metrics = ui_metric if type(ui_metric) == list else [ui_metric]
        self._dsm = dataset_sample_modifier

    def title(self) -> Optional[Union[str, List[str]]]:
        return [um.title() for um in self.ui_metrics]

    def description(self) -> Optional[Union[str, List[str]]]:
        return [um.description() for um in self.ui_metrics]

    def is_writable(self) -> bool:
        return any(um.is_writable() for um in self.ui_metrics)

    def update_metric(
        self, dataset_sample: Union[SequenceSample, SentencePairSample, TokensSample, QASample, str]
    ) -> None:
        new_sample = self._dsm(dataset_sample)
        for ui_metric in self.ui_metrics:
            ui_metric.update_metric(new_sample)

    def write_body(self) -> None:
        raise NotImplementedError

    def write_metric(self) -> None:
        for ui_metric in self.ui_metrics:
            if not ui_metric.is_writable():
                continue
            ui_metric.write_metric()


class LabelsUIMetric(UIMetric):
    """
    UIMetrics to compute the classes distribution in the dataset both counting them (histogram) and computing
    their frequency (pie chart).
    """

    def __init__(self):
        self.labels = []

    def title(self) -> str:
        return "Labels Occurrences"

    def description(self) -> Optional[Union[str, List[str]]]:
        return (
            "Statistics on the dataset labels. Histogram with total count of each class (Top). "
            "Pie chart with classes frequency (Bottom)."
        )

    def is_writable(self) -> bool:
        return len(self.labels) > 0

    def update_metric(self, dataset_sample: Union[SequenceSample, SentencePairSample, TokensSample]) -> None:
        sample_labels = dataset_sample.get_current_classification()
        if type(sample_labels) == str:
            sample_labels = [sample_labels]
        self.labels += sample_labels

    def write_body(self) -> None:
        labels_counter = collections.Counter(self.labels)
        x, y = zip(*[(lc, count) for lc, count in labels_counter.items()])
        classes_colors = [get_random_color() for _ in x]
        histogram = px.histogram(
            x=x,
            y=y,
            title="Classes total count",
            labels={"x": "classes", "y": "occurrences"},
            color_discrete_sequence=[classes_colors],
        )
        st.write(histogram)

        pie_chart = px.pie(
            names=x,
            values=y,
            title="Classes frequencies",
            labels={"names": "class", "values": "occurrences"},
            color_discrete_sequence=classes_colors,
        )
        st.write(pie_chart)


# === QA specific metrics ===
class AnswerPositionUIMetric(UIMetric):
    """
    UIMetric to compute the distribution of the position of the answers in the context.
    """

    def __init__(self):
        self._positions = []

    def title(self) -> Optional[Union[str, List[str]]]:
        return "Answer position in context"

    def description(self) -> Optional[Union[str, List[str]]]:
        return (
            "Avg, Min and Max position of the answer in the context. Expressed as the percentage of characters "
            "until the center of the answer. Boxplot and Histogram on the bottom."
        )

    def is_writable(self) -> bool:
        return len(self._positions) > 0

    def update_metric(self, dataset_sample: QASample) -> None:
        if dataset_sample.char_start is None or dataset_sample.char_end is None:
            return
        answer_center = round((dataset_sample.char_end + dataset_sample.char_start) / 2)
        self._positions.append(answer_center / len(dataset_sample.context) * 100)

    def write_body(self) -> None:
        c1, c2, c3 = st.columns(3)
        c1.metric(
            label=f"Avg answer position in context", value="{:.1f}%".format(sum(self._positions) / len(self._positions))
        )
        c2.metric(label=f"Min answer position in context", value="{:.1f}%".format(min(self._positions)))
        c3.metric(label=f"Max position in context", value="{:.1f}%".format(max(self._positions)))
        histogram = px.histogram(
            x=self._positions,
            nbins=10,
            labels={"x": "answer position in context"},
            color_discrete_sequence=[get_random_color()],
        )
        st.write(histogram)


def get_ui_metrics(task: str, tokenize: Optional[str]) -> List[UIMetric]:
    """
    Method that chooses the metrics to display based on the task and on the tokenization.
    Args:
        task: one of Sequence, Sentence-pair, Token and QA
        tokenize: the tokenizer language. Must be a valid language code for sacremoses. None means no tokenization

    Returns:
        the list of UIMetrics selected based on task and tokenization
    """
    if tokenize is not None:
        tokenizer = MosesTokenizer(lang=tokenize)

    ui_metrics = []

    if task == SEQUENCE:
        ui_metrics.append(
            LambdaWrapperUIMetric(
                InputLenUIMetric(
                    title="Sequence Characters",
                    description="Average, Min and Max sequences length in terms of characters (Top). "
                    "Quartiles on a boxplot (Bottom)",
                ),
                lambda sample: sample.sequence,
            )
        )
        ui_metrics.append(
            LambdaWrapperUIMetric(
                ClassSpecificInputLenUIMetric(
                    title_format='Sequence Characters for class "{}"',
                    description_format='Average, Min and Max sequences length for class "{}" '
                    "in terms of characters (Top). Quartiles on a boxplot (Bottom)",
                ),
                lambda sample: (sample.sequence, sample),
            )
        )
        if tokenize is not None:
            ui_metrics.append(
                LambdaWrapperUIMetric(
                    [
                        InputLenUIMetric(
                            title="Sequence Tokens",
                            description="Average, Min and Max sequences length in terms of tokens (Top). "
                            "Quartiles on a boxplot (Bottom)",
                        ),
                        ClassSpecificInputLenUIMetric(
                            title_format='Sequence Tokens for class "{}"',
                            description_format='Average, Min and Max sequences tokens for class "{}" in terms '
                            "of characters (Top). Quartiles on a boxplot (Bottom)",
                        ),
                    ],
                    lambda sample: (tokenizer.tokenize(sample.sequence), sample),
                )
            )
    elif task == SENTENCE_PAIR:
        # sentence 1 len in chars
        ui_metrics.append(
            LambdaWrapperUIMetric(
                InputLenUIMetric(
                    title="Sentence-1 Characters",
                    description="Average, Min and Max sentences length in terms of characters (Top). "
                    "Quartiles on a boxplot (Bottom)",
                ),
                lambda sample: sample.sentence1,
            )
        )
        # sentence 2 len in chars
        ui_metrics.append(
            LambdaWrapperUIMetric(
                InputLenUIMetric(
                    title="Sentence-2 Characters",
                    description="Average, Min and Max sentences length in terms of characters (Top). "
                    "Quartiles on a boxplot (Bottom)",
                ),
                lambda sample: sample.sentence2,
            )
        )
        if tokenize is not None:
            # sentence 1 len in tokens
            ui_metrics.append(
                LambdaWrapperUIMetric(
                    InputLenUIMetric(
                        title="Sentence-1 Tokens",
                        description="Average, Min and Max sentences length in terms of tokens (Top). "
                        "Quartiles on a boxplot (Bottom)",
                    ),
                    lambda sample: tokenizer.tokenize(sample.sentence1),
                )
            )
            # sentence 2 len in tokens
            ui_metrics.append(
                LambdaWrapperUIMetric(
                    InputLenUIMetric(
                        title="Sentence-2 Tokens",
                        description="Average, Min and Max sentences length in terms of tokens (Top). "
                        "Quartiles on a boxplot (Bottom)",
                    ),
                    lambda sample: tokenizer.tokenize(sample.sentence2),
                )
            )
    elif task == TOKEN:
        # input sequence len in chars (sequence = " ".join(tokens))
        ui_metrics.append(
            LambdaWrapperUIMetric(
                InputLenUIMetric(
                    title="Input Tokens Characters",
                    description="Average, Min and Max sequences length in terms of characters (Top). "
                    "Quartiles on a boxplot (Bottom). The sequences are computed by concatenating the "
                    "tokens with a white space.",
                ),
                lambda sample: " ".join(sample.tokens),
            )
        )
        # input sequence len in tokens
        ui_metrics.append(
            LambdaWrapperUIMetric(
                InputLenUIMetric(
                    title="Input Tokens",
                    description="Average, Min and Max sequences length in terms of tokens (Top). "
                    "Quartiles on a boxplot (Bottom)",
                ),
                lambda sample: sample.tokens,
            )
        )
    elif task == QA:
        # contexts len in chars
        ui_metrics.append(
            LambdaWrapperUIMetric(
                InputLenUIMetric(
                    title="Context Characters",
                    description="Average, Min and Max contexts length in terms of characters (Top). "
                    "Quartiles on a boxplot (Bottom)",
                ),
                lambda sample: sample.context,
            )
        )
        # questions len in chars
        ui_metrics.append(
            LambdaWrapperUIMetric(
                InputLenUIMetric(
                    title="Questions Characters",
                    description="Average, Min and Max questions length in terms of characters (Top). "
                    "Quartiles on a boxplot (Bottom)",
                ),
                lambda sample: sample.question,
            )
        )
        # answers len in chars
        ui_metrics.append(
            LambdaWrapperUIMetric(
                InputLenUIMetric(
                    title="Answer Characters",
                    description="Average, Min and Max answers length in terms of characters (Top). "
                    "Quartiles on a boxplot (Bottom)",
                ),
                lambda sample: sample.context[sample.char_start : sample.char_end]
                if sample.char_start is not None
                else None,
            )
        )

        if tokenize is not None:
            # contexts len in tokens
            ui_metrics.append(
                LambdaWrapperUIMetric(
                    InputLenUIMetric(
                        title="Context Tokens",
                        description="Average, Min and Max contexts length in terms of tokens (Top). "
                        "Quartiles on a boxplot (Bottom)",
                    ),
                    lambda sample: tokenizer.tokenize(sample.context),
                )
            )
            # questions len in tokens
            ui_metrics.append(
                LambdaWrapperUIMetric(
                    InputLenUIMetric(
                        title="Questions Tokens",
                        description="Average, Min and Max questions length in terms of tokens (Top). "
                        "Quartiles on a boxplot (Bottom)",
                    ),
                    lambda sample: tokenizer.tokenize(sample.question),
                )
            )
            # answers len in tokens
            ui_metrics.append(
                LambdaWrapperUIMetric(
                    InputLenUIMetric(
                        title="Answer Tokens",
                        description="Average, Min and Max answers length in terms of tokens (Top). "
                        "Quartiles on a boxplot (Bottom)",
                    ),
                    lambda sample: tokenizer.tokenize(sample.context[sample.char_start : sample.char_end])
                    if sample.char_start is not None
                    else None,
                )
            )
            ui_metrics.append(AnswerPositionUIMetric())

    else:
        print(f"ERROR: task {task} is not supported")
        raise NotImplementedError

    if task == SEQUENCE or task == SENTENCE_PAIR or task == TOKEN:
        ui_metrics.append(LabelsUIMetric())

    return ui_metrics


def update_metrics(ui_metrics: Iterable[UIMetric], task: str, dataset_path: str) -> None:
    data_driver = get_data_driver(task, dataset_path.split(".")[-1])
    dataset_samples = data_driver.read_from_path(dataset_path)
    for dataset_sample in dataset_samples:
        for ui_metric in ui_metrics:
            ui_metric.update_metric(dataset_sample)


def write_metrics(ui_metrics: Iterable[UIMetric]) -> None:
    for ui_metric in ui_metrics:
        if not ui_metric.is_writable():
            continue
        ui_metric.write_metric()


def init_layout(task: str, dataset_path: str, metrics: Iterable[UIMetric]):
    st.sidebar.markdown("# Describe")
    st.sidebar.markdown(
        f"""
            * **task**: {task.upper()}
            * **dataset path**: {dataset_path} 
        """
    )
    st.sidebar.markdown("### Metrics Index")
    metrics_count = 1
    for metric in metrics:
        titles = [metric.title()] if type(metric.title()) == str else metric.title()
        for title in titles:
            if type(title) != list:
                title = [title]
            for _t in title:
                st.sidebar.markdown(f"{metrics_count}. {_t}")
                metrics_count += 1

    st.title("Metrics Visualization")


def describe(task: str, dataset_path: str, tokenize: Optional[str]) -> None:
    """
    Main method - orchestrator
    Args:
        task: task name
        dataset_path: the dataset on which we run describe
        tokenize: whether to tokenize or not the input dataset and thus run additional metrics on the tokenized input,
        the value must be a language code supported by sacremoses

    Returns:
        None
    """
    ui_metrics = get_ui_metrics(task, tokenize)
    update_metrics(ui_metrics, task, dataset_path)
    init_layout(task, dataset_path, ui_metrics)
    write_metrics(ui_metrics)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("dataset")
    parser.add_argument("tokenize", nargs="?")
    return parser.parse_args()


def main():
    args = parse_args()
    describe(args.task, args.dataset, args.tokenize)


if __name__ == "__main__":
    main()
