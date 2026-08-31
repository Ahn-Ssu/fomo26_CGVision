import hydra
import os
import random
import torch
from asparagus.modules.transforms.presets import CPU_clsreg_val_test_transforms_crop
from asparagus.paths import get_config_path
from asparagus.pipeline.auto_configuration.checkpoint import load_checkpoint_state_dict
from asparagus.pipeline.auto_configuration.experiment_setup import prepare_inference
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

load_dotenv()

OmegaConf.register_new_resolver("random", lambda min, max: random.randint(min, max))
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(
    config_path=get_config_path(),
    config_name="default_test",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    file_store, path_store = prepare_inference(cfg)
    ckpt_cfg = OmegaConf.load(os.path.join(path_store.ckpt_parent_folder, "hydra/config.yaml"))
    output_path = os.path.join(
        path_store.ckpt_parent_folder,
        "predictions",
        cfg.test_task + "__" + cfg.test_split + "__" + cfg.load_checkpoint_name.replace(".ckpt", ".json"),
    )

    data_module = instantiate(
        ckpt_cfg.lightning._data_module,
        batch_size=1,
        train_split=None,
        val_split=None,
        test_samples=file_store.test,
        test_transforms=CPU_clsreg_val_test_transforms_crop(target_size=ckpt_cfg.training.target_size),
        num_workers=cfg.hardware.num_workers,
    )

    model = instantiate(
        ckpt_cfg.model._cls_net,
        input_channels=file_store.dataset_json.get("dataset_config", file_store.dataset_json["metadata"])["n_modalities"],
        output_channels=file_store.dataset_json.get("dataset_config", file_store.dataset_json["metadata"])["n_classes"],
    )

    model_module = instantiate(
        ckpt_cfg.lightning._lightning_module,
        model=model,
        weights=None,
        test_output_path=output_path,

    )
    checkpoint = torch.load(path_store.ckpt_path, map_location="cpu", weights_only=False)
    state_dict = {key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items() if key.startswith("model.")}
    incompatible = model_module.model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint restore mismatch: {incompatible}")
    print(f"Restored complete network checkpoint ({len(state_dict)} tensors): {path_store.ckpt_path}")


    trainer = instantiate(
        ckpt_cfg.lightning._trainer,
        callbacks=None,
        log_every_n_steps=250,
        logger=None,
        profiler=None,
        default_root_dir=path_store.run_dir,
        enable_progress_bar=True,
    )

    trainer.test(
        model=model_module,
        datamodule=data_module,
    )
    print(f"Test predictions saved to {output_path}")


if __name__ == "__main__":
    main()
