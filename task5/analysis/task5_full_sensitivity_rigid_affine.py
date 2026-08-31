import json, sys, os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/root/task5_submission_rigid')
sys.path.insert(0, '/root/task5_submission_rigid/model')
from net import ChampionTask1Net

from gardening_tools.modules.transforms.bias_field import Torch_BiasField
from gardening_tools.modules.transforms.blur import Torch_Blur
from gardening_tools.modules.transforms.motion_ghosting import Torch_MotionGhosting
from gardening_tools.modules.transforms.noise import Torch_AdditiveNoise, Torch_MultiplicativeNoise
from gardening_tools.modules.transforms.ringing import Torch_GibbsRinging
from gardening_tools.functional.transforms.sampling import torch_simulate_lowres
from gardening_tools.modules.transforms.spatial import Torch_Spatial
from gardening_tools.modules.transforms.mirror import Torch_Mirror
from monai.transforms import RandHistogramShift, RandScaleIntensity, RandRicianNoise, RandKSpaceSpikeNoise

SEED = 42
K_REPEATS = 5
axes3 = (0, 3)
TARGET_SIZE = (192, 224, 192)
EPS = 1e-7

MODELS = {
    "rigid":  {"data": "/root/asparagus_data/Task5_PMG_rigid_iso10", "fold_dir": "/root/task5_submission_rigid/model/fold_checkpoints"},
    "affine": {"data": "/root/asparagus_data/Task5_PMG_affine_iso10", "fold_dir": "/root/task5_submission_affine/model/fold_checkpoints"},
}

def apply_gamma_channel(ch, gamma):
    img_min, img_max = ch.min(), ch.max()
    img_range = img_max - img_min
    return torch.pow((ch - img_min) / (img_range + EPS), gamma) * (img_range + EPS) + img_min

def make_gamma5way(mode):
    invert = mode.startswith("inv")
    direction = None if mode == "inv_only" else ("bright" if "bright" in mode else "dark")
    gamma_range = None if direction is None else ((0.5, 0.999) if direction == "bright" else (1.0, 2.0))
    def _apply(x):
        out = x.clone()
        if invert:
            out = -out
        if direction is not None:
            gamma = np.random.uniform(*gamma_range)
            out[0, 0] = apply_gamma_channel(out[0, 0], gamma)
        return out
    return _apply

def make_lowres_axis(axis_idx):
    def _apply(x):
        out = x.clone()
        shape = np.array(out.shape[2:])
        zoom = np.random.uniform(0.5, 1.0)
        target_shape = shape.copy()
        target_shape[axis_idx] = max(1, int(round(shape[axis_idx] * zoom)))
        out[0, 0] = torch_simulate_lowres(out[0, 0], target_shape=tuple(target_shape), clip_to_input_range=False)
        return out
    return _apply

GPU_CONDITIONS = {
    "GPU_Blur":                lambda: (lambda x: Torch_Blur(p_per_channel=1.0)({"image": x})["image"]),
    "GPU_BiasField":           lambda: (lambda x: Torch_BiasField(p_per_channel=1.0)({"image": x})["image"]),
    "GPU_MotionGhosting":      lambda: (lambda x: Torch_MotionGhosting(p_per_channel=1.0, axes=axes3)({"image": x})["image"]),
    "GPU_GibbsRinging":        lambda: (lambda x: Torch_GibbsRinging(p_per_channel=1.0, axes=axes3)({"image": x})["image"]),
    "GPU_MultiplicativeNoise": lambda: (lambda x: Torch_MultiplicativeNoise(p_per_channel=1.0)({"image": x})["image"]),
    "GPU_AdditiveNoise":       lambda: (lambda x: Torch_AdditiveNoise(p_per_channel=1.0)({"image": x})["image"]),
    "GPU_LowRes_axis0_Sagittal": lambda: make_lowres_axis(0),
    "GPU_LowRes_axis1_Coronal":  lambda: make_lowres_axis(1),
    "GPU_LowRes_axis2_Axial":    lambda: make_lowres_axis(2),
    "GPU_Gamma_bright":     lambda: make_gamma5way("bright"),
    "GPU_Gamma_dark":       lambda: make_gamma5way("dark"),
    "GPU_Gamma_inv_bright": lambda: make_gamma5way("inv_bright"),
    "GPU_Gamma_inv_dark":   lambda: make_gamma5way("inv_dark"),
    "GPU_Gamma_inv_only":   lambda: make_gamma5way("inv_only"),
}

CPU_SPATIAL_CONDITIONS = {
    "CPU_Mirror_all": lambda: (lambda x: Torch_Mirror(
        p_per_sample=1.0, p_mirror_per_axis=1.0, axes=(0, 1, 2), skip_label=True
    )({"image": x})["image"]),
    "CPU_Rotation": lambda: (lambda x: Torch_Spatial(
        patch_size=TARGET_SIZE, crop=False, clip_to_input_range=False, skip_label=True,
        p_deform_all_channel=0.0, p_scale_all_channel=0.0,
        p_rot_all_channel=1.0, p_rot_per_axis=1.0,
        x_rot_in_degrees=(-15.0, 15.0), y_rot_in_degrees=(-15.0, 15.0), z_rot_in_degrees=(-15.0, 15.0),
    )({"image": x})["image"]),
    "CPU_Scale": lambda: (lambda x: Torch_Spatial(
        patch_size=TARGET_SIZE, crop=False, clip_to_input_range=False, skip_label=True,
        p_deform_all_channel=0.0, p_rot_all_channel=0.0,
        p_scale_all_channel=1.0, scale_factor=(0.85, 1.15),
    )({"image": x})["image"]),
}

class TorchMonaiIntensity:
    def __init__(self, monai_transform):
        self.monai_transform = monai_transform
    def __call__(self, x):
        out = self.monai_transform(x)
        return out.as_tensor() if hasattr(out, "as_tensor") else out

CPU_INTENSITY_CONDITIONS = {
    "CPU_RandHistogramShift": lambda: TorchMonaiIntensity(RandHistogramShift(num_control_points=(5, 15), prob=1.0)),
    "CPU_RandScaleIntensity": lambda: TorchMonaiIntensity(RandScaleIntensity(factors=0.1, prob=1.0)),
    "CPU_RandRicianNoise":    lambda: TorchMonaiIntensity(RandRicianNoise(prob=1.0, std=0.05, relative=True, sample_std=True)),
    "CPU_RandKSpaceSpikeNoise": lambda: TorchMonaiIntensity(RandKSpaceSpikeNoise(prob=1.0)),
}

DETERMINISTIC = {"CPU_Mirror_all"}
ALL_CONDITIONS = {**CPU_SPATIAL_CONDITIONS, **CPU_INTENSITY_CONDITIONS, **GPU_CONDITIONS}

device = torch.device('cuda')
net = ChampionTask1Net(input_channels=1, output_channels=2, teacher_name='brats', dropout_rate=0.3)
net.to(device); net.eval()

all_model_results = {}

for model_name, cfg in MODELS.items():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    split = json.load(open(f"{cfg['data']}/split_stratified5.json"))
    FOLD_DIR = cfg["fold_dir"]

    baseline_probs, baseline_labels = [], []
    cond_probs = {name: [] for name in ALL_CONDITIONS}
    cond_labels = {name: [] for name in ALL_CONDITIONS}

    with torch.no_grad():
        for fold in range(5):
            sd = torch.load(os.path.join(FOLD_DIR, f'fold{fold}.pt'), map_location=device, weights_only=True)
            net.load_state_dict(sd, strict=True)

            for p in split[fold]['val']:
                image, label = torch.load(p, map_location='cpu', weights_only=False)
                label = int(label.item())

                x0 = image.clone().unsqueeze(0).to(device)
                logits0 = net(x0)
                baseline_probs.append(F.softmax(logits0, dim=1)[0, 1].item())
                baseline_labels.append(label)

                for name, ctor in ALL_CONDITIONS.items():
                    fn = ctor()
                    k_reps = 1 if name in DETERMINISTIC else K_REPEATS
                    probs_k = []
                    for k in range(k_reps):
                        if name.startswith("CPU_"):
                            x_cpu = image.clone()
                            x_cpu = fn(x_cpu)
                            x = x_cpu.unsqueeze(0).to(device)
                        else:
                            x = image.clone().unsqueeze(0).to(device)
                            x = fn(x)
                        logits = net(x)
                        probs_k.append(F.softmax(logits, dim=1)[0, 1].item())
                    cond_probs[name].append(float(np.mean(probs_k)))
                    cond_labels[name].append(label)
        print(f"[{model_name}] all folds done", flush=True)

    base_auroc = roc_auc_score(baseline_labels, baseline_probs)
    res = {"baseline": base_auroc}
    for name in ALL_CONDITIONS:
        res[name] = roc_auc_score(cond_labels[name], cond_probs[name])
    all_model_results[model_name] = res

print()
print(f"seed={SEED}, K={K_REPEATS} (Mirror=1 pass)")
for model_name in MODELS:
    res = all_model_results[model_name]
    print(f"\n--- {model_name} (baseline={res['baseline']:.4f}) ---")
    rows = [(name, res[name], res[name] - res['baseline']) for name in ALL_CONDITIONS]
    for name, auroc, delta in sorted(rows, key=lambda r: r[2]):
        print(f"  {name:<28} {auroc:>10.4f} {delta:>+8.4f}")

print()
print("(참고: hdbet_only baseline=0.8333, GibbsRinging delta=-0.0417, MotionGhosting delta=-0.0365, BiasField delta=-0.0278)")
