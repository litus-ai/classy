import importlib
import os
import pkgutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Set

from argcomplete.completers import FilesCompleter

from classy.utils.experiment import Experiment, Run
from classy.utils.file import CLASSY_MODELS_CACHE_PATH


def get_device(device):
    if device == "gpu" or device == "cuda":
        return 0

    if device == "cpu":
        return -1

    try:
        return int(device)
    except ValueError:
        pass

    return device  # TODO: raise NotImplemented?


def autocomplete_model_path(prefix: str, **kwargs):
    # from argcomplete import warn

    if "/" in prefix:
        fc = FilesCompleter()

        # try standard file completion first
        tentative_files = fc(prefix)
        if tentative_files:
            # warn(tentative_files)
            return tentative_files

        # then iteratively go from most to least specific path
        # so we start when checkpoints is in the prefix
        if "checkpoints" in prefix:
            res = [
                name[len("experiments/") :]
                for name in fc("experiments/" + prefix)
                if name.endswith("ckpt")
            ]
            return res

        # maybe the user has entered
        tentative_path = Path(f"experiments/{prefix}/.hydra/config.yaml")
        if tentative_path.exists():
            run = Run.from_hydra_config(tentative_path)
            return [
                str(run.relative_directory / "checkpoints" / name)
                for name in run.checkpoint_names
            ]

        exp_name = prefix.split("/")[0]

        candidates = list(Experiment.from_name(exp_name).iter_candidate_runs(prefix))
        # warn(candidates)
        if len(candidates) > 1:
            return [str(run.relative_directory) for run in candidates]
        elif len(candidates) == 1:
            run = candidates[0]
            checkpoints = [
                str(run.relative_directory / "checkpoints" / name)
                for name in run.checkpoint_names
            ]
            # warn(checkpoints)
            checkpoints.insert(0, str(run.relative_directory))
            return checkpoints
        else:
            return []
    else:
        exps = (
            Experiment.list_available_experiments()
            + Experiment.list_downloaded_experiments()
        )

        # give the user the option to continue with a specific path of the experiment
        if os.path.exists("experiments") and os.path.isdir("experiments"):
            exps.append("experiments/")
        return exps


def checkpoint_path_from_user_input(model_path: str) -> str:
    try:
        path = try_get_checkpoint_path_from_user_input(model_path)
    except Exception as e:
        print(
            f"Unexpected exception occurred when converting your model argument"
            f" (`{model_path}`) to an actual checkpoint"
        )
        print("Exception:", e)
        print(
            "If you have trouble understanding where this may come from, open a new issue on GitHub."
        )
        print("https://github.com/sunglasses-ai/classy/issues/new")
        exit(1)

    if path is None:
        print(f"Unable to convert {model_path} to its actual checkpoint, exiting.")
        exit(1)

    return path


def try_get_checkpoint_path_from_user_input(model_path: str) -> Optional[str]:

    # immediately check if the path exists and is a checkpoint, in which case we return it
    if model_path.endswith(".ckpt") and Path(model_path).exists():
        return model_path

    # downloaded model!
    if "@" in model_path:
        exp = Experiment.from_name(model_path, CLASSY_MODELS_CACHE_PATH)

        if exp is None:
            print(
                f"No pretrained model called {model_path} was found! Available pretrained models:"
            )
            print(f"[{', '.join(Experiment.list_downloaded_experiments())}]")
            return None

        return str(exp.default_checkpoint)

    model_path = model_path.rstrip("/")
    model_name = (
        model_path[len("experiments/") :]
        if model_path.startswith("experiments/")
        else model_path
    )

    available_exps = Experiment.list_available_experiments()
    if model_name in available_exps:
        ckpt = Experiment.from_name(model_name).default_checkpoint
        return str(ckpt) if ckpt is not None else None

    p = Path("experiments") / model_name

    if p.exists() and p.name.endswith(".ckpt"):
        return str(p)

    # user actually selected a specific run
    tentative_path = p / ".hydra/config.yaml"
    if tentative_path.exists():
        ckpt = Run.from_hydra_config(tentative_path).default_checkpoint
        return str(ckpt) if ckpt is not None else None


@contextmanager
def push_python_path(path):
    """
    Prepends the given path to `sys.path`.
    This method is intended to use with `with`, so after its usage, its value willbe removed from
    `sys.path`.
    """
    # In some environments, such as TC, it fails when sys.path contains a relative path, such as ".".
    path = Path(path).resolve()
    path = str(path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        # Better to remove by value, in case `sys.path` was manipulated in between.
        sys.path.remove(path)


# from https://github.com/allenai/allennlp/blob/dcd8d9e9671da5a87de51f2bb42ceb3abdce8b3b/allennlp/common/util.py#L334
def import_module_and_submodules(
    package_name: str, exclude: Optional[Set[str]] = None
) -> None:
    """
    Import all submodules under the given package.
    Primarily useful so that people using classy as a library
    can specify their own custom packages and have their custom
    classes get loaded and registered.
    """
    if exclude and package_name in exclude:
        return

    importlib.invalidate_caches()

    with push_python_path("."):
        # Import at top level
        module = importlib.import_module(package_name)
        path = getattr(module, "__path__", [])
        path_string = "" if not path else path[0]

        # walk_packages only finds immediate children, so need to recurse.
        for module_finder, name, _ in pkgutil.walk_packages(path):
            # Sometimes when you import third-party libraries that are on your path,
            # `pkgutil.walk_packages` returns those too, so we need to skip them.
            if path_string and module_finder.path != path_string:
                continue
            subpackage = f"{package_name}.{name}"
            import_module_and_submodules(subpackage, exclude=exclude)


def maybe_find_directory(possible_names) -> Optional[str]:
    for possible_path in map(Path, possible_names):
        if possible_path.exists() and possible_path.is_dir():
            return str(possible_path)

    return None
