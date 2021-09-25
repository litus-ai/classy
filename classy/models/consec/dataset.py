from dataclasses import dataclass
from typing import Callable, Iterator, Union, Optional, NamedTuple, Tuple, List, Dict, Any

from classy.data.data_drivers import TokensSample, SequenceSample, SentencePairSample, QASample
from classy.data.dataset.base import BaseDataset, batchify, batchify_matrices
from classy.data.dataset.hf import HFBaseDataset
from classy.models.consec.disambiguation_corpus import DisambiguationInstance, DisambiguationCorpus
from classy.utils.vocabulary import Vocabulary


from dataclasses import dataclass
from typing import Callable, Iterator, List, NamedTuple, Dict, Any, Optional, Tuple, Iterable, Union

import numpy as np
import torch

from classy.models.consec.tokenizer import ConsecTokenizer
from classy.models.consec.dependency_finder import DependencyFinder
from classy.models.consec.sense_inventories import SenseInventory
from classy.utils.commons import flatten


class ConsecDefinition(NamedTuple):
    text: str
    linker: str  # it can be the instance lemma or text


@dataclass
class ConsecSample:
    token_sample: TokensSample  # needed in classy to backmap to token
    sample_id: str
    position: int  # position within disambiguation context
    disambiguation_context: List[DisambiguationInstance]
    candidate_definitions: List[ConsecDefinition]
    context_definitions: List[Tuple[ConsecDefinition, int]]  # definition and position within disambiguation context
    in_context_sample_id2position: Dict[str, int]
    disambiguation_instance: Optional[DisambiguationInstance] = None
    gold_definitions: Optional[List[ConsecDefinition]] = None
    marked_text: Optional[List[str]] = None  # this is set via side-effect
    kwargs: Optional[Dict[Any, Any]] = None

    def reset_context_definitions(self):
        self.marked_text = None
        self.context_definitions = []

    def add_context_definition(self, context_definition: ConsecDefinition, position: int):
        self.context_definitions.append((context_definition, position))

    def get_sample_id_position(self, sample_id: str) -> int:
        return self.in_context_sample_id2position[sample_id]


def build_samples_generator_from_disambiguation_corpus(
    sense_inventory: SenseInventory,
    disambiguation_corpus: Union[DisambiguationCorpus, List[DisambiguationCorpus]],
    dependency_finder: DependencyFinder,
    sentence_window: int,
    randomize_sentence_window: bool,
    remove_multilabel_instances: bool,
    shuffle_definitions: bool,
    randomize_dependencies: bool,
    sense_frequencies_path: Optional[str] = None,
) -> Callable[[], Iterator[ConsecSample]]:

    sense_frequencies = None
    if sense_frequencies_path is not None:
        sense_index = dict()
        senses_count = []
        with open(sense_frequencies_path) as f:
            for line in f:
                sense, count = line.strip().split("\t")
                sense_index[len(sense_index)] = sense
                senses_count.append(float(count))
        sense_frequencies = np.array(senses_count)
        sense_frequencies /= np.sum(sense_frequencies)

    def get_random_senses() -> List[str]:
        if sense_frequencies is None:
            return []

        n_senses = torch.distributions.Poisson(1).sample().item()
        if n_senses == 0:
            return []

        picked_senses_indices = np.random.choice(len(sense_index), int(n_senses), p=sense_frequencies, replace=False)

        picked_senses = [sense_index[psi] for psi in picked_senses_indices]

        return picked_senses

    def enlarge_disambiguation_context(
        disambiguation_context: List[DisambiguationInstance],
        instance_idx: int,
        dis_corpus: DisambiguationCorpus,
    ) -> Tuple[List[DisambiguationInstance], int]:

        prev_sent_num = next_sent_num = sentence_window // 2

        if randomize_sentence_window:
            # each randomization is independent
            prev_sent_num = int(torch.distributions.Poisson(prev_sent_num).sample().item())
            next_sent_num = int(torch.distributions.Poisson(next_sent_num).sample().item())

        disambiguation_instance = disambiguation_context[instance_idx]
        prev_sentences, next_sentences = dis_corpus.get_neighbours_sentences(
            disambiguation_instance.document_id, disambiguation_instance.sentence_id, prev_sent_num, next_sent_num
        )

        prev_disambiguation_instances = flatten(prev_sentences)
        next_disambiguation_instances = flatten(next_sentences)

        if len(prev_disambiguation_instances) > 0:
            instance_idx += len(prev_disambiguation_instances)

        return prev_disambiguation_instances + disambiguation_context + next_disambiguation_instances, instance_idx

    def shuffle_definitions_and_senses(definitions: List[str], senses: List[str]) -> Tuple[List[str], List[str]]:
        tmp_definitions_and_senses = list(zip(definitions, senses))
        np.random.shuffle(tmp_definitions_and_senses)
        definitions, senses = map(list, zip(*tmp_definitions_and_senses))
        return definitions, senses

    def get_randomized_context_senses_num(context_dependencies: List[DisambiguationInstance]) -> int:
        poisson_distr = torch.distributions.Poisson(1)
        sampled_percentage = (
            9.0 - poisson_distr.sample().item()
        ) / 9.0  # 9.0 is the maximum number reachable with poisson_lambda = 1

        sampled_num = round(sampled_percentage * len(context_dependencies))
        # sampled_num = int(poisson_distr.sample().item())

        return sampled_num

    def prepare_definitional_context(
        disambiguation_context: List[DisambiguationInstance], instance_idx: int
    ) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], List[str], Optional[List[str]]]:

        # Instance related
        disambiguation_instance = disambiguation_context[instance_idx]
        instance_possible_senses = (
            sense_inventory.get_possible_senses(disambiguation_instance.lemma, disambiguation_instance.pos)
            + get_random_senses()
        )

        if len(instance_possible_senses) == 0:
            print("Found an instance with no senses in the inventory: {}".format(disambiguation_instance))
            return None

        instance_possible_definitions = [sense_inventory.get_definition(sense) for sense in instance_possible_senses]

        if shuffle_definitions:
            instance_possible_definitions, instance_possible_senses = shuffle_definitions_and_senses(
                instance_possible_definitions, instance_possible_senses
            )

        # Context related
        context_ids, context_senses, context_lemmas, context_definitions, depends_from = [], [], [], [], []
        context_dependencies = dependency_finder.find_dependencies(disambiguation_context, instance_idx)
        num_dependencies_to_use = (
            get_randomized_context_senses_num(context_dependencies)
            if randomize_dependencies
            else len(context_dependencies)
        )

        if num_dependencies_to_use != 0:

            if num_dependencies_to_use != -1 and num_dependencies_to_use < len(context_dependencies):

                if randomize_dependencies:
                    context_dependencies_indices = np.random.choice(
                        list(range(len(context_dependencies))), num_dependencies_to_use, replace=False
                    )
                    context_dependencies = [context_dependencies[i] for i in sorted(context_dependencies_indices)]

                else:
                    context_dependencies = context_dependencies[:num_dependencies_to_use]

            for context_dependency in context_dependencies:
                dep_sense = context_dependency.labels[0]
                dep_definition = sense_inventory.get_definition(dep_sense)
                context_ids.append(context_dependency.instance_id)
                context_senses.append(dep_sense)
                context_lemmas.append(context_dependency.text)
                context_definitions.append(dep_definition)
                depends_from.append(context_dependency.instance_id)

        # Gold related
        gold_definitions = None
        if disambiguation_instance.labels is not None:
            gold_definitions = [
                definition
                for sense, definition in zip(instance_possible_senses, instance_possible_definitions)
                if sense in disambiguation_instance.labels
            ]

            if len(gold_definitions) == 0:
                return None

            if remove_multilabel_instances and len(gold_definitions) > 1:
                picked_gold_definition = np.random.choice(gold_definitions)

                filter_out_indices = {
                    idx
                    for idx, definition in enumerate(instance_possible_definitions)
                    if definition in gold_definitions and definition != picked_gold_definition
                }

                instance_possible_senses = [
                    sense for idx, sense in enumerate(instance_possible_senses) if idx not in filter_out_indices
                ]

                instance_possible_definitions = [
                    definition
                    for idx, definition in enumerate(instance_possible_definitions)
                    if idx not in filter_out_indices
                ]

                gold_definitions = [picked_gold_definition]

        return (
            instance_possible_senses,
            instance_possible_definitions,
            context_ids,
            context_senses,
            context_lemmas,
            context_definitions,
            depends_from,
            gold_definitions,
        )

    # MAIN METHOD
    def prepare_disambiguation_instance(
        disambiguation_context: List[DisambiguationInstance], instance_idx: int, dis_corpus: DisambiguationCorpus
    ) -> Optional[ConsecSample]:

        disambiguation_instance = disambiguation_context[instance_idx]

        if disambiguation_instance.instance_id is None:
            return None

        # consec_sample attributes will be stored here
        sample_store = dict(
            instance_id=disambiguation_instance.instance_id,
            instance_pos=disambiguation_instance.pos,
            instance_lemma=disambiguation_instance.lemma,
        )

        # === STEP-1: Enlarge disambiguation context
        # debugging purposes
        sample_store["original_disambiguation_context"] = disambiguation_context
        sample_store["original_disambiguation_index"] = instance_idx

        # step code
        disambiguation_context, instance_idx = enlarge_disambiguation_context(
            disambiguation_context, instance_idx, dis_corpus
        )
        sample_store["enlarged_disambiguation_context"] = disambiguation_context
        sample_store["enlarged_disambiguation_index"] = instance_idx

        sample_store["original_text"] = " ".join([di.text for di in disambiguation_context])  # debugging purposes

        # === STEP-2: Prepare definitional context
        # step code
        definitional_context = prepare_definitional_context(disambiguation_context, instance_idx)

        if definitional_context is None:
            return None

        (
            instance_possible_senses,
            instance_possible_definitions,  # instance related
            context_ids,
            context_senses,
            context_lemmas,
            context_definitions,
            depends_from,  # context instances related
            gold_definitions,  # gold related
        ) = definitional_context

        sample_store["context_definitions"] = context_definitions
        sample_store["context_senses"] = context_senses
        sample_store["depends_from"] = depends_from
        sample_store["instance_possible_definitions"] = instance_possible_definitions
        sample_store["instance_possible_senses"] = instance_possible_senses

        # build ConsecSample

        sample_id = disambiguation_instance.instance_id
        in_context_sample_id2position = {
            di.instance_id: i for i, di in enumerate(disambiguation_context) if di.instance_id is not None
        }

        candidate_consec_definitions = [
            ConsecDefinition(text=ipd, linker=disambiguation_instance.text.replace("_", " "))
            for ipd in instance_possible_definitions
        ]

        context_consec_definitions = [
            (ConsecDefinition(text=cd, linker=cl.replace("_", " ")), in_context_sample_id2position[cid])
            for cid, cd, cl in zip(context_ids, context_definitions, context_lemmas)
        ]

        gold_consec_definitions = []
        if gold_definitions is not None:
            gold_consec_definitions = [
                ConsecDefinition(text=igd, linker=disambiguation_instance.text.replace("_", " "))
                for igd in gold_definitions
            ]

        return ConsecSample(
            sample_id=sample_id,
            position=instance_idx,
            disambiguation_context=disambiguation_context,
            candidate_definitions=candidate_consec_definitions,
            context_definitions=context_consec_definitions,
            in_context_sample_id2position=in_context_sample_id2position,
            disambiguation_instance=disambiguation_instance,
            gold_definitions=gold_consec_definitions,
            token_sample=disambiguation_instance.token_sample,
            kwargs=sample_store,
        )

    # RETURNED FUNCTION
    def r() -> Iterator[ConsecSample]:

        disambiguation_corpora: List[DisambiguationCorpus] = (
            [disambiguation_corpus]
            if issubclass(disambiguation_corpus.__class__, DisambiguationCorpus)
            else disambiguation_corpus
        )

        done = [False for _ in disambiguation_corpora]
        iterators = [iter(d) for d in disambiguation_corpora]
        p = np.array([float(len(d)) for d in disambiguation_corpora])
        p /= np.sum(p)

        while True:

            if len(disambiguation_corpora) > 1:
                i = int(np.random.choice(len(disambiguation_corpora), 1, p=p)[0])
            else:
                i = 0

            try:
                disambiguation_context = next(iterators[i])
            except StopIteration:
                done[i] = True
                if all(done):
                    break
                iterators[i] = iter(disambiguation_corpora[i])
                disambiguation_context = next(iterators[i])

            for instance_idx in range(len(disambiguation_context)):
                consec_sample = prepare_disambiguation_instance(
                    disambiguation_context, instance_idx, disambiguation_corpora[i]
                )
                if consec_sample is not None:
                    yield consec_sample

    return r


class ConsecDataset(BaseDataset):
    def __init__(
        self,
        samples_iterator: Callable[[], Iterator[TokensSample]],
        tokenizer: ConsecTokenizer,
        use_definition_start: bool,
        text_encoding_strategy: str,
        is_doc_based: bool,
        sense_inventory: SenseInventory,
        dependency_finder: DependencyFinder,
        sentence_window: int,
        randomize_sentence_window: bool,
        remove_multilabel_instances: bool,
        shuffle_definitions: bool,
        randomize_dependencies: bool,
        sense_frequencies_path: Optional[str] = None,
        consec_samples_iterator: Callable[[], Iterator[TokensSample]] = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.use_definition_start = use_definition_start
        self.text_encoding_strategy = text_encoding_strategy
        self.sense_inventory = sense_inventory
        self.dependency_finder = dependency_finder
        self.sentence_window = sentence_window
        self.randomize_sentence_window = randomize_sentence_window
        self.remove_multilabel_instances = remove_multilabel_instances
        self.shuffle_definitions = shuffle_definitions
        self.randomize_dependencies = randomize_dependencies
        self.sense_frequencies_path = sense_frequencies_path

        # build consec samples iterator
        self.disambiguation_corpus = DisambiguationCorpus.from_samples(samples_iterator(), is_doc_based=is_doc_based)
        self.consec_samples_iterator = build_samples_generator_from_disambiguation_corpus(
            sense_inventory=self.sense_inventory,
            disambiguation_corpus=self.disambiguation_corpus,
            dependency_finder=self.dependency_finder,
            sentence_window=self.sentence_window,
            randomize_sentence_window=self.randomize_sentence_window,
            remove_multilabel_instances=self.remove_multilabel_instances,
            shuffle_definitions=self.shuffle_definitions,
            randomize_dependencies=self.randomize_dependencies,
            sense_frequencies_path=self.sense_frequencies_path,
        )

        super().__init__(samples_iterator=None, fields_batchers=None, batching_fields=["input_ids"], **kwargs)
        self._init_fields_batchers()

    @staticmethod
    def requires_vocab() -> bool:
        return False

    def _init_fields_batchers(self) -> None:
        self.fields_batcher = {
            "original_sample": None,  #
            "instance_id": None,  #
            "instance_pos": None,  #
            "instance_lemma": None,  #
            "input_ids": lambda lst: batchify(lst, padding_value=self.tokenizer.pad_token_id),  #
            "attention_mask": lambda lst: batchify(lst, padding_value=0),  #
            "token_type_ids": lambda lst: batchify(lst, padding_value=0),  #
            "original_disambiguation_context": None,  #
            "original_disambiguation_index": None,  #
            "enlarged_disambiguation_context": None,  #
            "enlarged_disambiguation_index": None,  #
            "instance_possible_definitions": None,  #
            "instance_possible_senses": None,  #
            "context_definitions": None,  #
            "context_senses": None,  #
            "depends_from": None,  #
            "definitions_mask": lambda lst: batchify(lst, padding_value=1),  #
            "definitions_offsets": None,  #
            "definitions_positions": None,  #
            "gold_senses": None,
            "gold_definitions": None,  #
            "gold_markers": lambda lst: batchify(lst, padding_value=0),  #
            "relative_positions": lambda lst: batchify_matrices(lst, padding_value=0),
        }

    def create_marked_text(self, sample: ConsecSample) -> List[str]:

        if self.text_encoding_strategy == "simple-with-linker" or self.text_encoding_strategy == "relative-positions":
            disambiguation_context = sample.disambiguation_context
            instance_idx = sample.position

            disambiguation_tokens = [di.text for di in disambiguation_context]
            marked_token = self.tokenizer.mark_token(
                disambiguation_tokens[instance_idx], marker=self.tokenizer.target_marker
            )
            disambiguation_tokens[instance_idx] = marked_token

            return disambiguation_tokens

        else:
            raise ValueError(f"Marking strategy {self.text_encoding_strategy} is undefined")

    def refine_definitions(
        self, sample: ConsecSample, definitions: List[ConsecDefinition], are_context_definitions: bool
    ) -> List[str]:

        if self.text_encoding_strategy == "simple-with-linker":

            # note: this is a direct coupling towards the tokenizer, which gets defined in a different independent yaml
            # file adding a safety assert -> if we are in this branch, tokenizer must have only 1 context_marker
            def_sep_token, def_end_token = self.tokenizer.context_markers[0]
            assert len(self.tokenizer.context_markers) == 1, (
                "Text encoding strategy is simple-with-linker, but multiple context markers, which would be unused, "
                "have been found. Conf error?"
            )

            return [
                f"{definition.capitalize()}. {def_sep_token} {linker} {def_end_token}"
                for definition, linker in definitions
            ]

        elif self.text_encoding_strategy == "relative-positions":
            def_sep_token, def_end_token = self.tokenizer.context_markers[0]
            assert len(self.tokenizer.context_markers) == 1, (
                "Text encoding strategy is simple-with-linker, but multiple context markers, which would be unused, "
                "have been found. Conf error?"
            )
            return [f"{def_sep_token} {definition.text.capitalize().strip('.')}." for definition in definitions]

        else:
            raise ValueError(f"Marking strategy {self.text_encoding_strategy} is undefined")

    def get_definition_positions(
        self, instance_possible_definitions: List[str], definitions_offsets: Dict[str, Tuple[int, int]]
    ) -> List[int]:
        definition_positions = []
        for definition in instance_possible_definitions:
            start_index, end_index = definitions_offsets[definition]
            running_index = start_index if self.use_definition_start else end_index
            definition_positions.append(running_index)
        return definition_positions

    @staticmethod
    def produce_definitions_mask(input_ids: torch.Tensor, definition_positions) -> torch.Tensor:
        definitions_mask = torch.ones_like(input_ids, dtype=torch.float)
        for definition_position in definition_positions:
            definitions_mask[definition_position] = 0.0
        return definitions_mask

    def produce_definition_markers(
        self, input_ids: torch.Tensor, gold_definitions: List[str], definitions_offsets: Dict[str, Tuple[int, int]]
    ) -> torch.Tensor:
        gold_markers = torch.zeros_like(input_ids)
        for definition in gold_definitions:
            start_index, end_index = definitions_offsets[definition]
            running_index = start_index if self.use_definition_start else end_index
            gold_markers[running_index] = 1.0
        return gold_markers

    def dataset_iterator_func(self) -> Iterable[Dict[str, Any]]:

        for sample in self.consec_samples_iterator():

            dataset_element = {"original_sample": sample, **sample.kwargs}

            # create marked text
            assert (
                sample.marked_text is None
            ), "Marked text is expected to be set via side-effect, but was found already set"
            sample.marked_text = self.create_marked_text(sample)

            # refine and text-encode definitions
            candidate_definitions = self.refine_definitions(
                sample, sample.candidate_definitions, are_context_definitions=False
            )
            context_definitions = self.refine_definitions(
                sample, [d for d, _ in sample.context_definitions], are_context_definitions=True
            )
            gold_definitions = (
                self.refine_definitions(sample, sample.gold_definitions, are_context_definitions=False)
                if sample.gold_definitions
                else None
            )

            # tokenize
            tokenization_out = self.tokenizer.tokenize(
                sample.marked_text,
                sample.get_sample_id_position(sample.sample_id),
                candidate_definitions,
                [(cd, pos) for cd, (_, pos) in zip(context_definitions, sample.context_definitions)],
            )
            input_ids, attention_mask, token_type_ids, definitions_offsets, relative_positions = tokenization_out
            dataset_element["input_ids"] = input_ids
            dataset_element["attention_mask"] = attention_mask
            dataset_element["definitions_offsets"] = definitions_offsets
            if token_type_ids is not None:
                dataset_element["token_type_ids"] = token_type_ids
            if relative_positions is not None:
                dataset_element["relative_positions"] = relative_positions

            # compute definitions position
            definition_positions = self.get_definition_positions(candidate_definitions, definitions_offsets)
            dataset_element["definitions_positions"] = definition_positions

            # compute definition mask
            definition_mask = self.produce_definitions_mask(input_ids, definition_positions)
            dataset_element["definitions_mask"] = definition_mask

            # create gold markers if present
            if gold_definitions is not None:
                dataset_element["gold_definitions"] = gold_definitions
                dataset_element["gold_markers"] = self.produce_definition_markers(
                    input_ids, gold_definitions, definitions_offsets
                )

            yield dataset_element
