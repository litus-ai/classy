import omegaconf
import hydra

import pytorch_lightning as pl

from classy.data.data_modules import ClassyDataModule
from classy.utils.hydra import fix
from classy.utils.vocabulary import FIELDS_VOCABULARY_PATH, LABELS_VOCABULARY_PATH


def train(conf: omegaconf.DictConfig) -> None:

    # reproducibility
    pl.seed_everything(conf.training.seed)

    # data module declaration
    pl_data_module: ClassyDataModule = hydra.utils.instantiate(conf.data.datamodule, _recursive_=False)
    pl_data_module.prepare_data()

    # main module declaration
    pl_module = hydra.utils.instantiate(conf.model, vocabulary=pl_data_module.vocabulary, _recursive_=False)

    # callbacks declaration
    callbacks_store = []

    # lightning callbacks
    if conf.training.early_stopping_callback is not None:
        early_stopping = hydra.utils.instantiate(conf.training.early_stopping_callback)
        callbacks_store.append(early_stopping)

    if conf.training.model_checkpoint_callback is not None:
        model_checkpoint = hydra.utils.instantiate(
            conf.training.model_checkpoint_callback,
            filename="{epoch:02d}-{" + conf.training.callbacks_monitor + ":.2f}",
        )
        callbacks_store.append(model_checkpoint)

    # model callbacks
    for callback in conf.callbacks.callbacks:
        callbacks_store.append(hydra.utils.instantiate(callback, _recursive_=False))

    # trainer
    trainer = hydra.utils.instantiate(
        conf.training.pl_trainer,
        callbacks=callbacks_store,
        **conf.device,
    )

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)


@hydra.main(config_path="../../../configurations/", config_name="root")
def main(conf: omegaconf.DictConfig):
    fix(conf)
    train(conf)


if __name__ == "__main__":
    main()