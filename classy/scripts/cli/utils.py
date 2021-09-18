from pathlib import Path

from argcomplete.completers import FilesCompleter
from classy.utils.experiment import Experiment, Run


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
    if "/" in prefix:
        fc = FilesCompleter()

        # try standard file completion first
        tentative_files = fc(prefix)
        if tentative_files:
            # warn(tentative_files)
            return tentative_files

        # then iteratively go from longest possible path to shortest
        # so we start when checkpoints is in the prefix
        if "checkpoints" in prefix:
            res = [name[len("experiments/") :] for name in fc("experiments/" + prefix) if name.endswith("ckpt")]
            # warn(fc("experiments/" + prefix))
            # warn([name.replace("experiments/", "") for name in fc("experiments/" + prefix)])
            # warn(res)
            return res

        tentative_path = Path(f"experiments/{prefix}/.hydra/config.yaml")
        if tentative_path.exists():
            run = Run.from_hydra_config(tentative_path)
            return [str(run.relative_directory / "checkpoints" / name) for name in run.checkpoint_names]

        exp_name = prefix.split("/")[0]

        candidates = Experiment.from_name(exp_name).iter_candidate_runs(prefix)
        candidates += [candidate + "/checkpoints/" for candidate in candidates]
        return candidates
    else:
        exps = Experiment.list_available_experiments()
        # TODO: list and add available downloaded models (without the path option! those only have one ckpt)
        # give the user the option to continue with a specific path of the experiment
        return exps + [name + "/" for name in exps]
