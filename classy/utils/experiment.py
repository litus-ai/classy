from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union, Iterator


class Experiment:
    def __init__(self, directory: Path):
        self._directory = directory

    @property
    def directory(self):
        return self._directory

    @classmethod
    def from_name(cls, exp_name: str, exp_dir: Optional[Union[str, Path]] = None) -> "Experiment":
        pass

    @classmethod
    def iter_experiments(cls, exps_dir: Optional[Union[str, Path]] = None) -> Iterator[Path]:
        exps_dir = cls.get_exp_dir(exps_dir)

        for exp_dir in exps_dir.iterdir():
            if next(exp_dir.rglob(".hydra/config.yaml"), None) is not None:
                yield exp_dir

    def iter_runs(self) -> Iterator["Run"]:
        pass  # TODO

    @classmethod
    def get_exp_dir(cls, exp_dir: Optional[Union[str, Path]] = None) -> Path:
        if exp_dir is None:
            p = Path.cwd() / "experiments"
        elif isinstance(exp_dir, Path):
            p = exp_dir
        elif isinstance(exp_dir, str):
            p = Path(exp_dir)
        else:
            raise NotImplementedError

        return p.absolute()


@dataclass
class Run:
    experiment: Experiment
    date: datetime
    directory: Path

    @property
    def hydra_dir(self) -> Path:
        return self.directory / ".hydra"

    @property
    def has_checkpoints(self):
        ckpt_dir = self.directory / "checkpoints"
        return ckpt_dir.exists() and len(list(ckpt_dir.iterdir())) > 0

    @property
    def checkpoints(self) -> Optional[List[Path]]:
        if not self.has_checkpoints:
            return None
        return list((self.directory / "checkpoints").iterdir())

    @property
    def checkpoint_names(self) -> Optional[List[str]]:
        ckpts = self.checkpoints
        if ckpts is None:
            return None
        return [ckpt.name for ckpt in ckpts]

    # TODO: should this return a DictConfig?
    @property
    def hydra_config(self) -> Path:
        return self.hydra_dir / "config.yaml"

    # TODO: should this return a DictConfig?
    @property
    def overrides(self) -> Path:
        return self.hydra_dir / "overrides.yaml"

    # TODO: should this return a DictConfig?
    @property
    def config(self) -> Path:
        return self.hydra_dir / "config.yaml"

    @classmethod
    def from_hydra_config(cls, hydra_config_path: Path, experiment: Experiment):
        directory = hydra_config_path.parent.parent
        date = datetime.strptime(f"{directory.parent.parent.name} {directory.parent.name}", "%Y-%m-%d %H-%M-%S")
        return cls(experiment, date, directory)
