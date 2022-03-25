import logging
import zipfile
from pathlib import Path
from typing import Optional

from classy.utils.experiment import Experiment, Run

logger = logging.getLogger(__name__)


def strip_checkpoint(
    checkpoint_path: Path,
    destination: Path,
    keys_to_remove=("callbacks", "optimizer_states", "lr_schedulers"),
):
    import torch

    logger.debug(f"loading checkpoint {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    for key in keys_to_remove:
        if ckpt.pop(key, None) is None:
            logger.debug(f"key {key} did not exist in checkpoint {checkpoint_path}")

    logger.debug(f"saving stripped checkpoint to {destination}")
    torch.save(ckpt, destination)


def zip_run(
    run: Run,
    tmpdir: Path,
    zip_name: str = "model.zip",
    strip_ckpt: bool = True,
    is_export: bool = False,
    best_only: bool = True,
) -> Path:

    logger.debug(f"zipping run {run} to {tmpdir}")
    # creates a zip version of the provided Run (with a single stripped checkpoint) in a model.zip file under `tmpdir`
    run_dir = run.directory
    ckpt_path = tmpdir / "best.ckpt"
    zip_path = tmpdir / zip_name

    relative_directory = run.experiment.directory.parent if is_export else run_dir

    with zipfile.ZipFile(zip_path, "w") as zip_file:

        # fully zip the run directory maintaining its structure
        for file in run_dir.rglob("*.*"):
            relative_name = file.relative_to(relative_directory)

            if file.is_dir():
                continue

            # skip checkpoints as we add a single checkpoint later
            if "checkpoints/" in str(relative_name):
                if best_only:
                    continue

                if strip_ckpt:
                    strip_checkpoint(file, ckpt_path)
                    zip_file.write(
                        ckpt_path, arcname=file.relative_to(relative_directory)
                    )
                    continue

            zip_file.write(file, arcname=file.relative_to(relative_directory))

        if best_only:
            ckpt_name = (
                run_dir.relative_to(relative_directory) / "checkpoints/best.ckpt"
            )

            if strip_ckpt:
                logger.debug("Stripping checkpoint before writing to zip file")
                strip_checkpoint(run.best_checkpoint, ckpt_path)
                logger.debug("Writing stripped checkpoint to zip file")
                zip_file.write(ckpt_path, arcname=ckpt_name)
            else:
                zip_file.write(run.best_checkpoint, arcname=ckpt_name)

        # remove stripped checkpoint file as it's inside the zip
        ckpt_path.unlink()

    return zip_path


def export(
    model_name: str,
    no_strip: bool,
    all_ckpts: bool,
    zip_name: Optional[str] = None,
):
    exp = Experiment.from_name(model_name)
    if exp is None:
        print(f"No experiment named {model_name} found. Exiting...")
        return

    run = exp.last_valid_run
    if run is None:
        print(f"No valid run found for experiment {model_name}. Exiting...")
        return

    zip_name = zip_name or f"classy-export-{model_name}.zip"

    zip_file = zip_run(
        run,
        Path.cwd(),
        zip_name=zip_name,
        strip_ckpt=not no_strip,
        is_export=True,
        best_only=not all_ckpts,
    )
    print(f"Model exported at {zip_file}")
