import hydra
import lightning as pl
import os
import random
from asparagus.functional.versioning import generate_unused_run_id
from asparagus.modules.hydra.plugins.searchpath_plugins import FinetuneSearchpathPlugin
from asparagus.modules.transforms.presets import CPU_clsreg_val_test_transforms_crop
from asparagus.paths import get_config_path
from asparagus.pipeline.auto_configuration.checkpoint import resolve_checkpoint
from asparagus.pipeline.auto_configuration.experiment_setup import (
    prepare_standard_experiment,
)
from asparagus.pipeline.auto_configuration.logging import logging
from dotenv import load_dotenv
from gardening_tools.modules.networks.components.weight_init import set_params_to_zero
from hydra.core.hydra_config import HydraConfig
from hydra.core.plugins import Plugins
from hydra.utils import instantiate
from asparagus.modules.callbacks import EmaAblationCallback, load_subjects
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from omegaconf import DictConfig, OmegaConf

load_dotenv()


OmegaConf.register_new_resolver("random", lambda min, max: random.randint(min, max))
OmegaConf.register_new_resolver(
    "version",
    lambda resume_training, run_dir: generate_unused_run_id(resume_training=resume_training, run_dir=run_dir),
    use_cache=True,
)
OmegaConf.register_new_resolver("eval", eval)
Plugins.instance().register(FinetuneSearchpathPlugin)


@hydra.main(
    config_path=get_config_path(),
    config_name="default_finetune_cls",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    print(f"{OmegaConf.to_yaml(cfg)}\n Version: {cfg.run_id}\n Run dir: {HydraConfig.get().run.dir}\n")
    logging_safe_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    file_store, path_store, version_store = prepare_standard_experiment(cfg)
    weights = resolve_checkpoint(cfg)
    pl.seed_everything(seed=cfg.training.seed, workers=True)

    loggers = logging(
        ckpt_wandb_id=version_store.wandb_id,
        ckpt_mlflow_id=version_store.mlflow_id,
        log_file_name=HydraConfig.get().job.name,
        run_dir=path_store.run_dir,
        version=version_store.version,
        wandb_config=logging_safe_cfg,
        wandb_experiment=HydraConfig.get().job.config_name,
        wandb_project=cfg.logger.wandb_project,
        wandb_logging=cfg.logger.wandb_logging,
        mlflow_logging=cfg.logger.mlflow_logging,
        log_to_stdout=cfg.logger.log_to_stdout,
    )

    best_ckpt_callback = ModelCheckpoint(
        dirpath=path_store.ckpt_save_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best",
        enable_version_counter=False,
    )
    # 2026-08-16: model.ckpt_every_n_steps (optional, default unset) switches
    # the periodic checkpoint callback from epoch-granularity to
    # step-granularity -- for tracking training dynamics finer than one
    # epoch (e.g. every 100 of 250 steps/epoch). Mutually exclusive with
    # every_n_epochs on a single ModelCheckpoint (Lightning constraint), so
    # branch on which cadence was requested; every_n_epochs stays the
    # default so existing epoch-based experiments are unaffected.
    if cfg.model.ckpt_every_n_steps:
        last_ckpt_callback = ModelCheckpoint(
            dirpath=path_store.ckpt_save_dir,
            every_n_train_steps=cfg.model.ckpt_every_n_steps,
            save_top_k=-1,
            # same auto_insert_metric_name behavior as the epoch case below --
            # bare placeholder gives "step=000100.ckpt", not "step=step=...".
            filename="{step:06d}",
            enable_version_counter=False,
        )
    else:
        last_ckpt_callback = ModelCheckpoint(
            dirpath=path_store.ckpt_save_dir,
            every_n_epochs=cfg.model.ckpt_every_n_epoch,
            save_top_k=-1,
            # ModelCheckpoint's default auto_insert_metric_name=True already
            # expands the `{epoch}` placeholder to "epoch=<value>" on its own --
            # a literal "epoch=" prefix in the template doubles it
            # ("epoch=epoch=00.ckpt", verified empirically 2026-08-15). Just the
            # bare placeholder gives the intended "epoch=00.ckpt".
            filename="{epoch:02d}",
            enable_version_counter=False,
        )

    progressbar_callback = TQDMProgressBar(refresh_rate=cfg.logger.log_every_n_steps)
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch", log_momentum=True)
    profilers = None

    cpu_tr_transforms = instantiate(
        cfg.transforms._cpu_tr_transforms,
        target_size=cfg.training.target_size,
        normalize=cfg.transforms.normalize,
    )
    cpu_val_transforms = instantiate(
        cfg.transforms._cpu_val_transforms,
        target_size=cfg.training.target_size,
        normalize=cfg.transforms.normalize,
    )
    gpu_tr_transforms = instantiate(cfg.transforms._gpu_tr_transforms, ndim=len(cfg.training.target_size))

    data_module = instantiate(
        cfg.lightning._data_module,
        train_split=file_store.splits["train"],
        val_split=file_store.splits["val"],
        train_transforms=cpu_tr_transforms,
        val_transforms=cpu_val_transforms,
        test_samples=file_store.test,
        test_transforms=CPU_clsreg_val_test_transforms_crop(
            target_size=cfg.training.target_size,
            normalize=cfg.transforms.normalize,
        ),
    )

    model = instantiate(
        cfg.model._cls_net,
        input_channels=file_store.dataset_json["metadata"]["n_modalities"],
        output_channels=file_store.dataset_json["metadata"]["n_classes"],
    )

    model_module = instantiate(
        cfg.lightning._lightning_module,
        model=model,
        warmup_epochs=(cfg.training.warmup_ratio * cfg.training.epochs if cfg.training.warmup_ratio is not None else cfg.training.warmup_epochs),
        decoder_warmup_epochs=cfg.training.decoder_warmup_epochs,
        train_transforms=gpu_tr_transforms,
        val_transforms=None,
        weights=weights,
        log_image_every_n_epochs=cfg.logger.log_images_every_n_epoch,
        optimizer=cfg.model.finetune_optim,
        learning_rate=cfg.model.finetune_lr,
        load_decoder=cfg.training.load_decoder,
        repeat_stem_weights=cfg.training.repeat_stem_weights,
        test_output_path=os.path.join(
            path_store.run_dir,
            "predictions",
            cfg.test_task + "__" + cfg.data.test_split + "__" + "best.json",
        ),
    )

    callbacks = [
        last_ckpt_callback,
        best_ckpt_callback,
        progressbar_callback,
        lr_monitor_callback,
    ]
    if cfg.model.ema_ablation:
        callbacks.append(
            EmaAblationCallback(
                betas=cfg.model.ema_betas,
                eval_every_n_steps=cfg.model.ckpt_every_n_steps,
                subjects=load_subjects(file_store.test),
                out_csv_path=os.path.join(path_store.run_dir, "predictions", "ema_ablation_by_step.csv"),
                ckpt_save_dir=os.path.join(path_store.run_dir, "ema_checkpoints"),
            )
        )

    trainer = instantiate(
        cfg.lightning._trainer,
        callbacks=callbacks,
        log_every_n_steps=cfg.logger.log_every_n_steps,
        logger=loggers,
        profiler=profilers,
        default_root_dir=path_store.run_dir,
        max_epochs=cfg.training.epochs,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        use_distributed_sampler=False,
    )

    trainer.fit(
        model=model_module,
        datamodule=data_module,
    )

    model_module.model.apply(set_params_to_zero)

    trainer.test(
        model=model_module,
        datamodule=data_module,
        ckpt_path=best_ckpt_callback.best_model_path,
    )


if __name__ == "__main__":
    main()
