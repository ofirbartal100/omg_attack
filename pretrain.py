'''Main pretraining script.'''
import warnings
warnings.filterwarnings('ignore')
import hydra

# pretrain_original
# pretrain_original_disc
@hydra.main(config_path='conf', config_name='ceva')
def run(config):
    # Deferred imports for faster tab completion
    import os
    import flatten_dict
    import pytorch_lightning as pl

    from dabs.src import online_evaluator
    #from dabs.src.datasets.catalog import MULTILABEL_DATASETS, PRETRAINING_DATASETS, UNLABELED_DATASETS
    from dabs.src.systems import emix, shed, viewmaker, viewmaker_original

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
        logger = pl.loggers.WandbLogger(entity="ofirbartal100", project='ceva-vm', name=config.exp.name)
    logger.log_hyperparams(flat_config)
    callbacks = [pl.callbacks.ModelCheckpoint(dirpath=save_dir,
                                              every_n_train_steps=config.trainer.get("model_checkpoint_freq", 20000),
                                              save_top_k=-1)]

    # assert config.dataset.name in PRETRAINING_DATASETS, f'{config.dataset.name} not one of {PRETRAINING_DATASETS}.'

    if config.algorithm == 'emix':
        system = emix.EMixSystem(config)
    elif config.algorithm == 'shed':
        system = shed.ShEDSystem(config)
    elif config.algorithm == 'viewmaker':
        system = viewmaker.ViewmakerSystem(config)
    elif config.algorithm == 'original_viewmaker':
        system = viewmaker_original.OriginalViewmakerSystem(config)
    elif config.algorithm == 'viewmaker_original_disc':
        system = viewmaker_original.ViewmakerOriginalSystemDisc(config)
    elif config.algorithm == 'viewmaker_ceva_disc':
        system = viewmaker_original.CevaViewmakerSystem(config)
    elif config.algorithm == 'conditional_viewmaker':
        system = viewmaker_original.ConditionalViewmakerSystem(config)
    elif config.algorithm == 'tama_viewmaker':
        system = viewmaker_original.ViewmakerTAMA38System(config)
    elif config.algorithm == 'viewmaker_disc':
        system = viewmaker.ViewmakerSystemDisc(config)
    elif config.algorithm == 'traffic_viewmaker':
        system = viewmaker_original.TrafficViewMaker(config)
    elif config.algorithm == 'birds_viewmaker':
        system = viewmaker_original.BirdsViewMaker(config)
    else:
        raise ValueError(f'Unimplemented algorithm config.algorithm={config.algorithm}.')

    # Online evaluator for labeled datasets.
    #if config.dataset.name not in UNLABELED_DATASETS:
    #    ssl_online_evaluator = online_evaluator.SSLOnlineEvaluator(
    #        dataset=config.dataset.name,
    #        z_dim=config.model.kwargs.dim,
    #        num_classes=system.dataset.num_classes(),
    #        multi_label=(config.dataset.name in MULTILABEL_DATASETS),
    #    )
    #    callbacks += [ssl_online_evaluator]

    # PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        default_root_dir=save_dir,
        logger=logger,
        gpus=config.gpus,  # GPU indices
        max_steps=config.trainer.max_steps,
        min_steps=config.trainer.max_steps,
        resume_from_checkpoint=config.trainer.resume_from_checkpoint if config.start_enc > 0 else None,
        val_check_interval=config.trainer.val_check_interval,
        limit_val_batches=config.trainer.limit_val_batches,
        callbacks=callbacks,
        profiler="simple",
        #weights_summary=config.trainer.weights_summary,
        gradient_clip_val=config.trainer.gradient_clip_val,
        precision=config.trainer.precision,
        strategy=config.trainer.distributed_backend or None,
    )

    #if config.dataset.name not in UNLABELED_DATASETS:
    #    ssl_online_evaluator.on_pretrain_routine_start(trainer,system)
        
    trainer.fit(system)


if __name__ == '__main__':
    run()

