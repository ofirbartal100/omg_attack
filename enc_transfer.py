'''Main transfer script.'''

import hydra
import os


@hydra.main(config_path='conf', config_name='transfer_enc')
def run(config):
    # Deferred imports for faster tab completion
    import os

    import flatten_dict
    import pytorch_lightning as pl

    from dabs.src.datasets.catalog import TRANSFER_DATASETS
    from dabs.src.systems import enc_transfer

    pl.seed_everything(config.trainer.seed)

    # Saving checkpoints and logging with wandb.
    flat_config = flatten_dict.flatten(config, reducer='dot')
    save_dir = os.path.join(config.exp.base_dir, config.exp.name)

    # override dataset configs if needed
    if config.get("num_workers") is not None:
        config.dataset.num_workers = config.num_workers

    # set logger
    if config.get("debug", False):
        config.dataset.num_workers = 0
        logger = pl.loggers.TensorBoardLogger(save_dir="tensorboard", name=config.exp.name)
    else:
        logger = pl.loggers.WandbLogger(entity="shafir", project='domain-agnostic-transfer', name=config.exp.name)
    logger.log_hyperparams(flat_config)
    ckpt_callback = pl.callbacks.ModelCheckpoint(dirpath=save_dir)

    # assert config.dataset.name in TRANSFER_DATASETS, f'{config.dataset.name} not one of {TRANSFER_DATASETS}.'

    # PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        default_root_dir=save_dir,
        logger=logger,
        gpus=config.gpus,
        max_epochs=config.trainer.max_epochs,
        min_epochs=config.trainer.max_epochs,
        val_check_interval=config.trainer.val_check_interval,
        limit_val_batches=config.trainer.limit_val_batches,
        callbacks=[ckpt_callback],
        weights_summary=config.trainer.weights_summary,
        precision=config.trainer.precision
    )

    # system = transfer.TransferSystem(config)
    system = enc_transfer.TransferSystem(config)
    trainer.fit(system)


if __name__ == '__main__':
    run()
