from abc import ABC
from typing import List, Tuple, Callable

import numpy as np
import torch

from examples.consec.disambiguation_corpus import DisambiguationInstance
from examples.consec.sense_inventories import SenseInventory
from examples.consec.utils import pos_map


class DependencyFinder(ABC):
    def __init__(self, max_dependencies: int = -1):
        self.max_dependencies = max_dependencies

    def find_dependencies(
        self, disambiguation_context: List[DisambiguationInstance], instance_idx: int
    ) -> List[DisambiguationInstance]:
        dependencies = self._find_dependencies(disambiguation_context, instance_idx)
        if self.max_dependencies >= 0:
            dependencies = dependencies[: self.max_dependencies]
        return dependencies

    def _find_dependencies(
        self, disambiguation_context: List[DisambiguationInstance], instance_idx: int
    ) -> List[DisambiguationInstance]:
        raise NotImplementedError


class EmptyDependencyFinder(DependencyFinder):
    def _find_dependencies(
        self, disambiguation_context: List[DisambiguationInstance], instance_idx: int
    ) -> List[DisambiguationInstance]:
        return []


class PolysemyDependencyFinder(DependencyFinder):
    def __init__(self, sense_inventory: SenseInventory, max_dependencies: int = -1):
        super().__init__(max_dependencies)
        self.sense_inventory = sense_inventory

    def _find_dependencies(
        self, disambiguation_context: List[DisambiguationInstance], instance_idx: int
    ) -> List[DisambiguationInstance]:
        polysemy_ordered_instances = sorted(
            [di for di in disambiguation_context if di.instance_id is not None],
            key=lambda di: len(self.sense_inventory.get_possible_senses(di.lemma, di.pos)),
        )
        instances_id = [di.instance_id for di in polysemy_ordered_instances]
        current_instance_id = instances_id.index(disambiguation_context[instance_idx].instance_id)
        return polysemy_ordered_instances[:current_instance_id]


class PPMIPolysemyDependencyFinder(PolysemyDependencyFinder):
    def __init__(
        self,
        sense_inventory: SenseInventory,
        single_counter_path: str,
        pair_counter_path: str,
        energy: float,
        max_dependencies: int = -1,
        normalize_ppmi: bool = False,
        minimum_ppmi: float = 0.0,
        undirected: bool = False,
        with_pos: bool = True,
    ):
        super().__init__(sense_inventory, max_dependencies)
        self.energy = energy
        self.normalize_ppmi = normalize_ppmi
        self.minimum_ppmi = minimum_ppmi
        self.undirected = undirected
        self.with_pos = with_pos
        self.ppmi_func = self.setup_ppmi_func(single_counter_path, pair_counter_path)

    def setup_ppmi_func(
        self,
        single_counter_path: str,
        pair_counter_path: str,
    ) -> Callable[[Tuple[str, str], Tuple[str, str]], float]:
        def split_lp(lp):
            if self.with_pos:
                l = lp[: lp.rindex(".")]
                p = lp[lp.rindex(".") + 1 :]
                p = pos_map.get(p, p)
            else:
                l = lp
                p = "FAKE-POS"
            return l, p

        # read single counter
        single_counter = {}
        N = 0
        with open(single_counter_path) as f:
            for line in f:
                lp, c = line.strip().split("\t")
                l, p = split_lp(lp)
                c = int(float(c))
                single_counter[(l, p)] = c
                N += c

        # read pair counter
        pair_counter = {}
        with open(pair_counter_path) as f:
            for line in f:
                lp1, lp2, c = line.strip().split("\t")
                l1, p1 = split_lp(lp1)
                l2, p2 = split_lp(lp2)
                assert (l1, p1) in single_counter and (l2, p2) in single_counter, f"{(l1, p1)} | {(l2, p2)}"
                pair_counter[((l1, p1), (l2, p2))] = int(float(c))

        def f(k1: Tuple[str, str], k2: Tuple[str, str]) -> float:
            try:
                pxy = (pair_counter[k1, k2] if (k1, k2) in pair_counter else pair_counter[k2, k1]) / N
                px = single_counter[k1] / N
                py = single_counter[k2] / N
                sample_pmi = np.log2(pxy / (px * py))
                if self.normalize_ppmi:
                    sample_pmi /= -np.log2(pxy)
            except KeyError:
                sample_pmi = 0.0
            return max(sample_pmi, 0.0)

        return f

    def score_dependencies(
        self, disambiguation_context: List[DisambiguationInstance], instance_idx: int
    ) -> Tuple[List[DisambiguationInstance], torch.Tensor]:
        def di2lp(di: DisambiguationInstance):
            if self.with_pos:
                return di.lemma, di.pos
            else:
                return di.text.replace(" ", "_"), "FAKE-POS"

        x = disambiguation_context[instance_idx]
        if self.undirected:
            instance_dependencies = [
                dep for i, dep in enumerate(disambiguation_context) if i != instance_idx and dep.instance_id is not None
            ]
        else:
            instance_dependencies = super()._find_dependencies(disambiguation_context, instance_idx)

        # compute ppmi for each dependency
        ppmis = torch.tensor([self.ppmi_func(di2lp(x), di2lp(y)) for y in instance_dependencies])

        # threshold on ppmis if minimum_ppmi is set
        ppmis[ppmis < self.minimum_ppmi] = 0.0

        return instance_dependencies, ppmis

    def _find_dependencies(
        self, disambiguation_context: List[DisambiguationInstance], instance_idx: int
    ) -> List[DisambiguationInstance]:

        instance_dependencies, ppmis = self.score_dependencies(disambiguation_context, instance_idx)

        if all(score == 0.0 for score in ppmis):
            return []

        # convert to probability
        ps = ppmis / ppmis.sum()

        # take indices up to self.energy cumulative probability
        indices = []
        cp = 0.0
        for index in ps.argsort(descending=True):

            if ps[index] == 0.0:
                break

            cp += ps[index]
            indices.append(index)

            if cp > self.energy:
                break

        return [instance_dependencies[i] for i in indices]
