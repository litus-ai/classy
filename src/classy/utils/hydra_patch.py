import copy
import sys
from dataclasses import dataclass
from typing import Any, List

from hydra._internal.config_loader_impl import ConfigLoaderImpl
from hydra._internal.config_repository import IConfigRepository
from hydra.core.default_element import ResultDefault
from hydra.core.override_parser.types import Override
from hydra.errors import ConfigCompositionException
from hydra.plugins.config_source import ConfigResult
from omegaconf import DictConfig, ListConfig, OmegaConf, ValidationError, flag_override


class ConfigBlame:
    def __init__(self):
        raise ValueError(
            "Cannot instantiate this directly. Either call from_override or from_default_and_result!"
        )

    @classmethod
    def from_default_and_result(cls, default: ResultDefault, result: ConfigResult):
        return NormalConfigBlame(default, result)

    @classmethod
    def from_override(cls, override: Override):
        return OverrideConfigBlame(override)

    def __str__(self):
        raise NotImplementedError

    def __hash__(self):
        return hash(self.__str__())


@dataclass
class NormalConfigBlame(ConfigBlame):
    _PROVIDER_SHORT = {
        "classy-searchpath-plugin": "classy",
    }

    default: ResultDefault
    result: ConfigResult

    def __post_init__(self):
        # if we keep it, hydra has troubles resolving the root config
        self.result.config = None

    def __str__(self):
        provider = NormalConfigBlame._PROVIDER_SHORT.get(
            self.result.provider, self.result.provider
        )
        return f"[source: {provider}/{self.default.config_path}]"

    def __hash__(self):
        return hash(str(self))


@dataclass
class OverrideConfigBlame(ConfigBlame):
    override: Override

    def __str__(self):
        return f"[override: {self.override.input_line}]"

    def __hash__(self):
        return hash(str(self))


def _compose_config_from_defaults_list_patch(
    self,
    defaults: List[ResultDefault],
    repo: IConfigRepository,
) -> DictConfig:
    cfg = OmegaConf.create()
    blame = []

    with flag_override(cfg, "no_deepcopy_set_nodes", True):
        for default in defaults:
            loaded = self._load_single_config(default=default, repo=repo)
            loaded: ConfigResult

            try:
                cfg.merge_with(loaded.config)

                if loaded.provider == "hydra":
                    continue

                cfg_blame = (
                    flatten_keys(loaded.config),
                    ConfigBlame.from_default_and_result(default, loaded),
                )

                blame.append(cfg_blame)

            except ValidationError as e:
                raise ConfigCompositionException(
                    f"In '{default.config_path}': Validation error while composing config:\n{e}"
                ).with_traceback(sys.exc_info()[2])

    cfg.__dict__["_blame"] = blame
    return cfg


def _apply_overrides_to_config(overrides: List[Override], cfg: DictConfig) -> None:
    ConfigLoaderImpl._apply_overrides_to_config_orig(overrides, cfg)
    blame = cfg.__dict__.get("_blame")

    for override in overrides:
        blame.append(([override.key_or_group], ConfigBlame.from_override(override)))
        # print(override.input_line, override.key_or_group, override.type, override.value_type, override.package)


def dict_config_deepcopy_patch(self, memo):
    res = DictConfig.__deepcopy_orig__(self, memo)
    blame = self.__dict__.get("_blame")

    if blame is not None:
        res.__dict__["_blame"] = copy.deepcopy(blame, memo=memo)

    return res


DictConfig.__deepcopy_orig__ = DictConfig.__deepcopy__
DictConfig.__deepcopy__ = dict_config_deepcopy_patch


def flatten_keys(cfg: Any, resolve: bool = False) -> List[str]:
    ret = []

    def handle_container(key: Any, value: Any, resolve: bool) -> List[str]:
        return [f"{key}.{k}" for k in flatten_keys(value, resolve=resolve)]

    def handle_iterator(iterator):
        for k, v in iterator:
            append_key = True

            if isinstance(v, (DictConfig, ListConfig)):
                to_extend = handle_container(k, v, resolve=resolve)
                ret.extend(to_extend)
                append_key = len(to_extend) == 0

            if append_key:
                ret.append(str(k))

    if isinstance(cfg, DictConfig):
        handle_iterator(cfg.items_ex(resolve=resolve))

    elif isinstance(cfg, ListConfig):
        handle_iterator(enumerate(cfg._iter_ex(resolve=resolve)))

    else:
        assert False

    return ret


ConfigLoaderImpl._apply_overrides_to_config_orig = (
    ConfigLoaderImpl._apply_overrides_to_config
)
ConfigLoaderImpl._apply_overrides_to_config = _apply_overrides_to_config
ConfigLoaderImpl._compose_config_from_defaults_list = (
    _compose_config_from_defaults_list_patch
)
