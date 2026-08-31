import numpy as np
import os
import sys
import tempfile
import nibabel as nib
import torch
from asparagus.modules.transforms.presets import CPU_clsreg_val_test_transforms_crop
from asparagus.pipeline.auto_configuration.checkpoint import load_checkpoint_state_dict
from dotenv import load_dotenv
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf
from torch.nn.functional import softmax

sys.path.insert(0, "/root/FOMO26")
from data.normalize import asparagus_volume_wise_znorm
from data.preprocess_raw import _sitk_resample
import SimpleITK as sitk

load_dotenv()

MODEL_DIR = None
CHECKPOINT_NAME = None


def preprocess_task1_niftis(data: list, target_size: list | tuple) -> torch.Tensor:
    """Match the iso1mm Task-1 training preprocessing for raw NIfTIs."""
    if len(data) != 4 or any(path is None for path in data):
        raise ValueError("Task 1 requires FLAIR, ADC, DWI_b1000, and exactly one of SWI/T2*.")

    images = [nib.load(path) for path in data]
    reference = images[0]
    for path, image in zip(data[1:], images[1:]):
        if image.shape != reference.shape:
            raise ValueError(f"Modality shape mismatch: {path}: {image.shape} != {reference.shape}")
        if not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-5):
            raise ValueError(f"Modality affine mismatch: {path}")

    resampled, masks = [], []
    for image in images:
        array = image.get_fdata().astype(np.float32)
        if not np.isfinite(array).all():
            raise ValueError("Task 1 input contains NaN or Inf.")
        spacing = [float(s) for s in image.header.get_zooms()[:3]]
        native_mask = (array > 1e-6).astype(np.float32)
        array_iso = _sitk_resample(
            array, spacing, target_spacing=(1.0, 1.0, 1.0),
            interpolator=sitk.sitkBSpline, default_value=0.0,
        )
        mask_iso = _sitk_resample(
            native_mask, spacing, target_spacing=(1.0, 1.0, 1.0),
            interpolator=sitk.sitkNearestNeighbor, default_value=0.0,
        ) > 0.5
        resampled.append(np.where(mask_iso, array_iso, 0.0).astype(np.float32))
        masks.append(mask_iso)

    shapes = {array.shape for array in resampled}
    if len(shapes) != 1:
        raise ValueError(f"Resampled modality shapes disagree: {sorted(shapes)}")
    union_mask = np.logical_or.reduce(masks)
    if not union_mask.any():
        raise ValueError("No foreground found in Task 1 input.")
    coords = np.argwhere(union_mask)
    mins, maxs = coords.min(axis=0), coords.max(axis=0) + 1
    crop_slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))

    channels = []
    for array, mask in zip(resampled, masks):
        cropped, cropped_mask = array[crop_slices], mask[crop_slices]
        if not cropped_mask.any():
            raise ValueError("A modality has no foreground inside the union crop.")
        channels.append(asparagus_volume_wise_znorm(cropped, mask=cropped_mask))

    target_size = tuple(int(v) for v in target_size)
    crop_shape = channels[0].shape
    if any(size > target for size, target in zip(crop_shape, target_size)):
        raise ValueError(f"Foreground crop {crop_shape} exceeds target size {target_size}.")
    output = np.zeros((4, *target_size), dtype=np.float32)
    starts = [(target - size) // 2 for size, target in zip(crop_shape, target_size)]
    dst = tuple(slice(start, start + size) for start, size in zip(starts, crop_shape))
    for channel_idx, channel in enumerate(channels):
        output[(channel_idx, *dst)] = channel
    return torch.from_numpy(output)


def main(
    data: list,
    output_path: str,
    output_channels: int,
    input_channels: int,
    checkpoint_dir: str,
    checkpoint_name: str,
    accelerator: str,
) -> None:
    ckpt_cfg = OmegaConf.load(os.path.join(checkpoint_dir, "hydra/config.yaml"))
    output_path = output_path
    image = preprocess_task1_niftis(data, ckpt_cfg.training.target_size)
    temporary_dir = tempfile.TemporaryDirectory(prefix="fomo26_task1_predict_")
    tensor_path = os.path.join(temporary_dir.name, "task1_input.pt")
    torch.save(image, tensor_path)

    data_module = instantiate(
        ckpt_cfg.lightning._data_module,
        batch_size=1,
        train_split=None,
        val_split=None,
        predict_samples=[tensor_path],
        predict_transforms=CPU_clsreg_val_test_transforms_crop(
            target_size=ckpt_cfg.training.target_size,
            normalize=ckpt_cfg.transforms.normalize,
        ),
        num_workers=0,
    )

    model = instantiate(
        ckpt_cfg.model._cls_net,
        input_channels=input_channels,
        output_channels=output_channels,
    )

    model_module = instantiate(
        ckpt_cfg.lightning._lightning_module,
        model=model,
        weights=load_checkpoint_state_dict(os.path.join(checkpoint_dir, f"checkpoints/{checkpoint_name}.ckpt")),
        test_output_path=output_path,
    )

    trainer = Trainer(accelerator=accelerator)

    output = trainer.predict(
        model=model_module,
        datamodule=data_module,
        return_predictions=True,
    )

    output = softmax(output[0], dim=1)  # Get the probability of the positive class
    np.savetxt(output_path, output[:, 1].cpu().numpy())
    temporary_dir.cleanup()

    print(f"Test predictions saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict script for FOMO26 Task 1: Infarct Detection")
    parser.add_argument("--flair", type=str, required=True)
    parser.add_argument("--adc", type=str, required=True)
    parser.add_argument("--dwi", type=str, required=True)
    parser.add_argument("--t2s", type=str, required=False, help="Path to T2* image (optional)")
    parser.add_argument("--swi", type=str, required=False, help="Path to SWI image (optional)")
    parser.add_argument("--output", type=str, required=True, help="Path to save predictions")
    parser.add_argument("--input_channels", type=int, default=4, help="Number of input channels for the model")
    parser.add_argument("--output_channels", type=int, default=2, help="Number of output channels for the model")
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Accelerator to use for prediction (e.g., 'cpu', 'cuda', 'mps')",
    )
    args = parser.parse_args()

    if args.t2s is not None:
        data = [args.flair, args.adc, args.dwi, args.t2s]
    else:
        data = [args.flair, args.adc, args.dwi, args.swi]

    assert MODEL_DIR is not None, "MODEL_DIR environment variable must be set to the path of the model checkpoint directory"
    assert CHECKPOINT_NAME is not None, (
        "CHECKPOINT_NAME environment variable must be set to the name of the checkpoint file (without .pt extension)"
    )

    main(
        data=data,
        output_path=args.output,
        output_channels=args.output_channels,
        input_channels=args.input_channels,
        checkpoint_dir=MODEL_DIR,
        checkpoint_name=CHECKPOINT_NAME,
        accelerator=args.accelerator,
    )
