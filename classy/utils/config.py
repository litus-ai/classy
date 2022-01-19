from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from omegaconf import DictConfig, ListConfig, OmegaConf

from classy.utils.hydra_patch import ConfigBlame


class ConfigBlamer:
    def __init__(self, blame):
        self.key2blame = self.process_blame_data(blame)

        self._empty = object()
        self._cache = {}

        # for key in sorted(self.key2blame, key=lambda k: k.count("."), reverse=True):
        for key in list(self.key2blame):
            self.blame(key)

        it = (k for k in self.iter_all() if k not in self.key2blame)
        for key in sorted(it, key=lambda e: e.count("."), reverse=True):
            self.solve_unseen(key)

    @staticmethod
    def _expand(key):
        parts = key.split(".")
        return [".".join(parts[: i + 1]) for i in range(len(parts))]

    def iter_all(self):
        seen = set()
        for key in self.key2blame:
            for k in self._expand(key):
                if k in seen:
                    continue
                yield k
                seen.add(k)

    def children(self, key):
        return [
            item
            for item in self.key2blame
            if item.startswith(key) and item.count(".") - 1 == key.count(".")
        ]

    def solve_unseen(self, key):
        children = self.children(key)
        self.key2blame[key] = self.get_most_common_blame(children[0])

    def blame(self, key):
        if key not in self.key2blame:
            self.key2blame[key] = self.compute_blame(key)

        return self.key2blame.get(key)

    @staticmethod
    def parent(key):
        return ".".join(key.split(".")[:-1])

    @staticmethod
    def direct_descendant(key, parent):
        return parent in key and key[len(parent) + 1 :].count(".") == 0

    def compute_blame(self, key) -> str:
        direct_blame = self.key2blame.get(key, self._empty)

        if direct_blame is self._empty:
            blamed = self.get_most_common_blame(key)
            self.key2blame[key] = blamed
            return blamed

        return direct_blame

    def get_most_common_blame(self, key):
        blames = self.gather_blames(key)
        return max(blames.keys(), key=lambda e: blames[e], default=None)

    def should_show_blame(self, key):
        parent = self.parent(key)
        return self.blame(parent) != self.blame(key)

    def gather_blames(self, key):
        blames = {}

        for k, v in self.key2blame.items():

            if not k.startswith(key):
                continue

            if not self.direct_descendant(k, key):
                continue

            blames[v] = blames.get(v, 0) + 1

        return blames

    @staticmethod
    def process_blame_data(blame_data: Tuple[List[str], ConfigBlame]):
        key2blame = {}

        for keys, config_blame in blame_data:
            for key in keys:
                key2blame[key] = config_blame

        return key2blame


@dataclass
class NodeInfo:
    key: str
    _expl: "ExplainableConfig"

    @property
    def interpolation(self) -> Optional[str]:
        interpolation = self._expl.try_get_interpolation(self.key)
        if interpolation is None:
            return None

        return str(interpolation)

    @property
    def is_leaf(self):
        return self._expl.is_leaf(self.key)

    @property
    def value(self) -> Any:
        return OmegaConf.select(self._expl.cfg, self.key)

    @property
    def blame(self) -> Optional[str]:
        if not self._expl.should_show_blame(self.key):
            return None

        return self._expl.blame(self.key)


class ExplainableConfig:
    def __init__(
        self,
        config: DictConfig,
        additional_blames: List[Tuple[List[str], ConfigBlame]] = (),
    ):
        self.cfg = config

        blame = config.__dict__.pop("_blame", None)
        self.has_blames = blame is not None

        # TODO: should we print a warning here?
        # assert (
        #     blame is not None
        # ), "`config.__dict__['_blame']` is None. are you using this class after importing `classy.utils.hydra_patch`?"

        self._blame = ConfigBlamer((blame or []) + (additional_blames or []))

    @staticmethod
    def split_key(key):
        *parents, k = key.split(".")
        return ".".join(parents), k

    def try_get_interpolation(self, key):
        parent, k = self.split_key(key)
        obj = OmegaConf.select(self.cfg, parent)
        if isinstance(obj, ListConfig):
            k = int(k)

        if OmegaConf.is_interpolation(obj, k):
            node = obj.__dict__["_content"][k]
            interpolation_value = node._value()
            # resolved_value = node._dereference_node()._value()

            return interpolation_value
        else:
            return None

    def get_node_info(self, key) -> NodeInfo:
        return NodeInfo(key, self)

    def is_leaf(self, key):
        return not OmegaConf.is_config(OmegaConf.select(self.cfg, key))

    def should_show_blame(self, key):
        return self.has_blames and self._blame.should_show_blame(key)

    def blame(self, key):
        return self._blame.blame(key)
