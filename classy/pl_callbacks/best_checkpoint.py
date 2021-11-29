import shutil
from pathlib import Path

from pytorch_lightning.callbacks import ModelCheckpoint


class ModelCheckpointWithBest(ModelCheckpoint):
    """
    A callback that explicitly saves the best checkpoint with best.ckpt.
    Note that the best checkpoint is duplicated, rather than linked, in best.ckpt
    """

    CHECKPOINT_NAME_BEST = "best.ckpt"

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        if self.best_model_path == "":
            return
        orig_best = Path(self.best_model_path)
        shutil.copyfile(orig_best, orig_best.parent / self.CHECKPOINT_NAME_BEST)
