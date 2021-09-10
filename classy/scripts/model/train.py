import omegaconf
import hydra

import pytorch_lightning as pl

from classy.data.data_modules import ClassyDataModule
from classy.utils.hydra import fix


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

    # logging
    logger = None

    # wandb
    if conf.logging.wandb.use_wandb:
        from pytorch_lightning.loggers import WandbLogger

        wandb_params = dict(
            project=conf.logging.wandb.project_name,
            name=conf.logging.wandb.experiment_name,
        )
        if conf.logging.wandb.anonymous is not None:
            wandb_params["anonymous"] = "allow"

        logger = WandbLogger(**wandb_params)

    # trainer
    trainer: pl.trainer.Trainer = hydra.utils.instantiate(
        conf.training.pl_trainer,
        callbacks=callbacks_store,
        logger=logger,
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
