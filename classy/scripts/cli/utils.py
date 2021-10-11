import os

from pathlib import Path
from typing import Optional

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
            res = [name[len("experiments/") :] for name in fc("experiments/" + prefix) if name.endswith("ckpt")]
            return res

        # maybe the user has entered
        tentative_path = Path(f"experiments/{prefix}/.hydra/config.yaml")
        if tentative_path.exists():
            run = Run.from_hydra_config(tentative_path)
            return [str(run.relative_directory / "checkpoints" / name) for name in run.checkpoint_names]

        exp_name = prefix.split("/")[0]

        candidates = list(Experiment.from_name(exp_name).iter_candidate_runs(prefix))
        # warn(candidates)
        if len(candidates) > 1:
            return [str(run.relative_directory) for run in candidates]
        elif len(candidates) == 1:
            run = candidates[0]
            checkpoints = [str(run.relative_directory / "checkpoints" / name) for name in run.checkpoint_names]
            # warn(checkpoints)
            checkpoints.insert(0, str(run.relative_directory))
            return checkpoints
        else:
            return []
    else:
        exps = Experiment.list_available_experiments() + Experiment.list_downloaded_experiments()

        # give the user the option to continue with a specific path of the experiment
        if os.path.exists("experiments") and os.path.isdir("experiments"):
            exps.append("experiments/")
        return exps


def checkpoint_path_from_user_input(model_path: str) -> str:
    path = try_get_checkpoint_path_from_user_input(model_path)
    if path is None:
        print(f"Unable to convert {model_path} to its actual checkpoint, exiting.")
        exit(1)

    return path


def try_get_checkpoint_path_from_user_input(model_path: str) -> Optional[str]:

    # downloaded model!
    if "@" in model_path:
        exp = Experiment.from_name(model_path, CLASSY_MODELS_CACHE_PATH)

        if exp is None:
            print(f"No pretrained model called {model_path} was found! Available pretrained models:")
            print(f"[{', '.join(Experiment.list_downloaded_experiments())}]")
            return None

        return str(exp.default_checkpoint)

    model_path = model_path.rstrip("/")
    model_name = model_path[len("experiments/") :] if model_path.startswith("experiments/") else model_path

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
