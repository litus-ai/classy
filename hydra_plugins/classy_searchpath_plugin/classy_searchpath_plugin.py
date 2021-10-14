from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class ClassySearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # from https://hydra.cc/docs/next/advanced/search_path/#creating-a-searchpathplugin
        search_path.append(provider="classy-searchpath-plugin", path="pkg://configurations")
