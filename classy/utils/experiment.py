from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Union

from classy.utils.file import CLASSY_MODELS_CACHE_PATH


class Experiment:
    def __init__(self, directory: Path):
        self._directory = directory

    @property
    def directory(self):
        return self._directory

    @property
    def name(self):
        return self.directory.name

    @property
    def last_run(self) -> "Run":
        return max(self.iter_runs(), key=lambda r: r.date, default=None)

    @property
    def has_checkpoint(self):
        return any(run.has_checkpoints for run in self.iter_runs())

    def last_run_by(self, fn) -> Optional["Run"]:
        return max(filter(fn, self.iter_runs()), key=lambda r: r.date, default=None)

    @property
    def default_checkpoint(self):
        return (
            self.last_run_by(
                lambda r: r.best_checkpoint is not None
            )  # first look for a run that has a best.ckpt
            or self.last_run_by(
                lambda r: r.last_checkpoint is not None
            )  # if it fails, find last available checkpoint
        ).default_checkpoint  # this gives precedence to best, then sorts by last available

    def __str__(self):
        return f"Experiment<{self.name}>(runs={sum(1 for _ in self.iter_runs())})"

    def __repr__(self):
        return str(self)

    @classmethod
    def from_name(
        cls, exp_name: str, exp_dir: Optional[Union[str, Path]] = None
    ) -> Optional["Experiment"]:
        exp_dir = cls.get_exp_dir(exp_dir)
        directory = exp_dir / exp_name

        if not directory.exists():
            return None

        return Experiment(directory)

    @classmethod
    def iter_experiments(
        cls, exps_dir: Optional[Union[str, Path]] = None
    ) -> Iterator["Experiment"]:
        exps_dir = cls.get_exp_dir(exps_dir)

        if not exps_dir.exists():
            return []

        for exp_dir in exps_dir.iterdir():
            if next(exp_dir.rglob(".hydra/config.yaml"), None) is not None:
                yield Experiment(exp_dir)

    @classmethod
    def list_available_experiments(
        cls, exps_dir: Optional[Union[str, Path]] = None
    ) -> List[str]:
        return list(
            ex.name for ex in cls.iter_experiments(exps_dir) if ex.has_checkpoint
        )

    @classmethod
    def list_downloaded_experiments(cls) -> List[str]:
        return cls.list_available_experiments(CLASSY_MODELS_CACHE_PATH)

    def iter_runs(self) -> Iterator["Run"]:
        for config_file in self.directory.rglob(".hydra/config.yaml"):
            yield Run.from_hydra_config(hydra_config_path=config_file, experiment=self)

    def iter_candidate_runs(self, prefix: Optional[str] = None) -> Iterator["Run"]:
        valid_runs = filter(lambda r: r.has_checkpoints, self.iter_runs())
        if prefix is not None:
            valid_runs = filter(
                lambda r: prefix in str(r.relative_directory), valid_runs
            )

        yield from valid_runs

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
    def num_checkpoints(self) -> int:
        ckpts = self.checkpoints
        if ckpts is None:
            return 0
        return len(ckpts)

    @property
    def default_checkpoint(self) -> Optional[Path]:
        if not self.has_checkpoints:
            return None

        return self.best_checkpoint or self.last_checkpoint

    @property
    def best_checkpoint(self) -> Optional[Path]:
        best = self.directory / "checkpoints/best.ckpt"
        return best if best.exists() else None

    @property
    def last_checkpoint(self) -> Path:
        return max(self.checkpoints, key=lambda p: p.stat().st_ctime)

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

    @property
    def relative_directory(self):
        return self.directory.relative_to(self.experiment.directory.parent)

    def __str__(self):
        date = self.date.strftime("%Y/%m/%d %H:%M:%S")
        return f"Run(exp={self.experiment.name}, date={date}, num_ckpt={self.num_checkpoints})"

    def __repr__(self):
        return str(self)

    @classmethod
    def from_hydra_config(
        cls, hydra_config_path: Path, experiment: Optional[Experiment] = None
    ):
        # path structure is experiments/experiment-name/YYYY-MM-DD/HH-mm-ss/.hydra/config.yaml
        #          which is experiments/  parents[3]   /parents[2]/parents[1]/parents[0] when using Path.parents
        parents = hydra_config_path.parents
        date = datetime.strptime(
            f"{parents[2].name} {parents[1].name}", "%Y-%m-%d %H-%M-%S"
        )
        experiment = experiment or Experiment(parents[3])
        return cls(experiment, date, parents[1])
