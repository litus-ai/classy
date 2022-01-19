import os

try:
    import rich
    from rich.console import Console
    from rich.style import Style
    from rich.text import Text
    from rich.tree import Tree
except ImportError:
    print("classy train [...] --print requires `pip install rich`")
    exit()

from typing import Iterable, List, Optional, Tuple, Union

from omegaconf import DictConfig, ListConfig, OmegaConf

from classy.utils.config import ExplainableConfig, NodeInfo
from classy.utils.hydra_patch import ConfigBlame, NormalConfigBlame


class RichNodeInfo:
    _CLASSY_GITHUB_CONFIG_URL = (
        "https://github.com/sunglasses-ai/classy/tree/main/configurations"
    )

    def __init__(self, info: NodeInfo):
        self.info = info

    def render_value(self) -> Text:
        value = self.info.value

        if value is None:
            return Text.from_markup(
                "None",
                style=Style(
                    bold=True,
                    color="orange1",
                ),
            )

        if value is True:
            return Text.from_markup(
                "True",
                style=Style(
                    bold=True,
                    color="green",
                ),
            )

        if value is False:
            return Text.from_markup(
                "False",
                style=Style(
                    bold=True,
                    color="red",
                ),
            )

        if isinstance(value, (int, float)):
            return Text(str(value), style=Style(color="cyan"))

        return Text(value, style=Style(color="hot_pink"))

    def __rich__(self):

        key_name = self.info.key.split(".")[-1]

        parts = [
            key_name,
        ]

        if self.info.is_leaf:
            value = self.render_value()
            parts.append(": ")
            parts.append(value)
        else:
            value = self.info.value
            if len(value) == 0:
                if OmegaConf.is_list(value):
                    v = "[]"
                elif OmegaConf.is_dict(value):
                    v = "{}"
                else:
                    raise ValueError(
                        f"key {self.info.key} is neither a dict nor a list. {value}"
                    )

                parts.append(": ")
                parts.append(Text(v, style=Style(bold=True, color="yellow3")))

        interp = self.info.interpolation
        if interp:
            parts.append(Text(f" [interp: {interp}]", style=Style(color="magenta")))

        blame = self.info.blame
        if blame:
            if isinstance(blame, NormalConfigBlame):
                # TODO: maybe we can improve this?
                blame = str(blame)
                assert blame.startswith("[source: ") and blame.endswith(
                    "]"
                ), f"Unknown blame: {blame}"
                blame_val = blame[len("[source: ") : -1]
                provider, config = blame_val.split("/", 1)
                if provider == "classy":
                    # if rich.console
                    config_url = (
                        f"{RichNodeInfo._CLASSY_GITHUB_CONFIG_URL}/{config}.yaml"
                    )
                    parts.append(
                        Text.assemble(
                            " [source: ",
                            Text.from_markup(
                                f"[link={config_url}][blue]classy/{config}[/blue][/link]",
                                style="blue",
                            ),
                            "]",
                            style="blue",
                        )
                    )
            else:
                parts.append(Text(f" {blame}", style=Style(color="blue")))

        return Text.assemble(*parts)


class ConfigPrinter:
    def __init__(
        self,
        cfg: Union[dict, DictConfig],
        fields_order: Iterable[str] = (
            "training",
            "model",
            "data",
            "prediction",
            "callbacks",
            "logging",
        ),
        skip_remaining: bool = False,
        additional_blames: Optional[List[Tuple[List[str], "ConfigBlame"]]] = None,
    ):
        self.expl = ExplainableConfig(cfg, additional_blames)
        self.fields_order = fields_order
        self.skip_remaining = skip_remaining

    def get_rich_tree(self) -> Tree:
        style = "dim"
        tree = Tree("<root>", guide_style=style)

        ordered_keys = list(self.fields_order)

        if not self.skip_remaining:
            ordered_keys += sorted(
                set(self.expl.cfg.keys()).difference(self.fields_order)
            )

        for key in ordered_keys:
            for branch in self.walk_config(key, sort=False):
                tree.add(branch)

        return tree

    @staticmethod
    def join_keys(parent: Optional[str], key: str):
        if parent is None:
            return key

        return f"{parent}.{key}"

    def walk_config(self, key, sort: bool = True) -> List[Tree]:
        sort_fn = sorted if sort else lambda item: item
        info = self.expl.get_node_info(key)
        value = info.value

        t = Tree(RichNodeInfo(info))

        if not info.is_leaf:
            iterator = None

            if isinstance(value, (DictConfig, dict)):
                iterator = sort_fn(value.keys())

            if isinstance(value, (ListConfig, list)):
                iterator = map(str, range(len(value)))

            assert iterator is not None, f"{key}: {value} is neither a List nor a Dict"

            for k in iterator:
                for child in self.walk_config(self.join_keys(key, k)):
                    t.add(child)

        return [t]


def get_rich_tree_config(
    cfg: DictConfig, blames: Optional[List[Tuple[List[str], ConfigBlame]]] = None
):
    return ConfigPrinter(cfg, additional_blames=blames).get_rich_tree()


def print_config(
    cfg: DictConfig, blames: Optional[List[Tuple[List[str], ConfigBlame]]] = None
):
    rich.print(get_rich_tree_config(cfg, blames))


RICH_ST_CODE_FORMAT = (
    "<pre style=\"font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace; "
    'line-height: 1.1; background-color: rgb(248, 249, 251); ">'
    "<code>{code}</code>"
    "</pre>"
)


def rich_to_html(renderable, print_to_console: bool = False, width: int = 230):
    with open(os.devnull, "w") as f:
        console = Console(
            record=True, file=None if print_to_console else f, width=width
        )
        console.print(renderable)
        html = console.export_html(inline_styles=True, code_format=RICH_ST_CODE_FORMAT)

        # adjust links to open in new windows
        html = html.replace("<a ", '<a target="_blank" ')

        return html
