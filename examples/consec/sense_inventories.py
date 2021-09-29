import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List

from nltk.corpus import wordnet as wn

from examples.consec.utils import pos_map


logger = logging.getLogger(__name__)


class SenseInventory(ABC):
    @abstractmethod
    def get_possible_senses(self, lemma: str, pos: str) -> List[str]:
        pass

    @abstractmethod
    def get_definition(self, sense: str) -> str:
        pass


# WORDNET


@lru_cache(maxsize=None)
def gloss_from_sense_key(sense_key: str) -> str:
    return wn.lemma_from_key(sense_key).synset().definition()


class WordNetSenseInventory(SenseInventory):
    def __init__(self, wn_candidates_path: str):
        self.lemmapos2senses = dict()
        self._load_lemmapos2senses(wn_candidates_path)

    def _load_lemmapos2senses(self, wn_candidates_path: str):
        logger.debug("Loading WordNet sense inventory")
        with open(wn_candidates_path) as f:
            for line in f:
                lemma, pos, *senses = line.strip().split("\t")
                self.lemmapos2senses[(lemma, pos)] = senses
        logger.debug("WordNet sense inventory loaded")

    def get_possible_senses(self, lemma: str, pos: str) -> List[str]:
        return self.lemmapos2senses.get((lemma, pos), [])

    def get_definition(self, sense: str) -> str:
        return gloss_from_sense_key(sense)


class XLWSDSenseInventory(SenseInventory):
    def __init__(self, inventory_path: str, definitions_path: str):
        self.lemmapos2synsets = dict()
        self._load_inventory(inventory_path)
        self.synset2definition = dict()
        self._load_synset_definitions(definitions_path)

    def _load_inventory(self, inventory_path: str) -> None:
        with open(inventory_path) as f:
            for line in f:
                lemmapos, *synsets = line.strip().split("\t")
                lemma, pos = lemmapos.split("#")
                pos = pos_map[pos]
                self.lemmapos2synsets[(lemma, pos)] = synsets

    def _load_synset_definitions(self, definitions_path: str) -> None:
        with open(definitions_path) as f:
            for line in f:
                synset, definition = line.strip().split("\t")
                self.synset2definition[synset] = definition

    def get_possible_senses(self, lemma: str, pos: str) -> List[str]:
        return self.lemmapos2synsets.get((lemma.lower().replace(" ", "_"), pos), [])

    def get_definition(self, sense: str) -> str:
        return self.synset2definition[sense]
