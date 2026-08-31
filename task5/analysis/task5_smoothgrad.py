import sys, os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, '/root/task5_submission_hdbet_only')
sys.path.insert(0, '/root/task5_submission_hdbet_only/model')
from net import ChampionTask1Net

device = torch.device('cuda')
net = ChampionTask1Net(input_channels=1, output_channels=2, teacher_name='brats', dropout_rate=0.3)
net.to(device)
net.eval()

FOLD_DIR = '/root/task5_submission_hdbet_only/model/fold_checkpoints'
sd = torch.load(os.path.join(FOLD_DIR, 'fold1.pt'), map_location=device, weights_only=True)
net.load_state_dict(sd, strict=True)

SUBJECTS = {
    "sub_25_PMG_positive": "/root/asparagus_data/Task5_PMG_hdbet_only_iso07/preprocessed/sub_25/ses_01/t1.pt",
    "sub_04_control_negative": "/root/asparagus_data/Task5_PMG_hdbet_only_iso07/preprocessed/sub_04/ses_01/t1.pt",
}

N_SMOOTH = 30
NOISE_FRAC = 0.08  # noise sigma as a fraction of the image's own std

def compute_smoothgrad(image):
    img_std = image.std().item()
    sigma = NOISE_FRAC * img_std
    sal_accum = None
    prob_accum = 0.0
    for i in range(N_SMOOTH):
        noisy = image + torch.randn_like(image) * sigma
        x = noisy.clone().unsqueeze(0).to(device)
        x.requires_grad_(True)
        logits = net(x)
        prob = F.softmax(logits, dim=1)[0, 1]
        net.zero_grad()
        prob.backward()
        grad = x.grad.detach()[0, 0]
        sal = (grad * x.detach()[0, 0]).abs()
        sal_accum = sal if sal_accum is None else sal_accum + sal
        prob_accum += prob.item()
    return prob_accum / N_SMOOTH, sal_accum.cpu().numpy() / N_SMOOTH

fig, axes = plt.subplots(len(SUBJECTS), 6, figsize=(24, 4 * len(SUBJECTS)))

for row, (name, path) in enumerate(SUBJECTS.items()):
    image, label = torch.load(path, map_location='cpu', weights_only=False)
    label = int(label.item())
    prob, sal_np = compute_smoothgrad(image)
    img_np = image[0].numpy()
    print(f"{name}: true_label={label}, p(PMG) (mean over {N_SMOOTH} noisy passes)={prob:.4f}")

    X, Y, Z = img_np.shape
    sal_clip = np.clip(sal_np, 0, np.percentile(sal_np, 99.5))
    sal_norm = sal_clip / (sal_clip.max() + 1e-8)

    slices = [
        ("sagittal", img_np[X // 2, :, :], sal_norm[X // 2, :, :]),
        ("coronal", img_np[:, Y // 2, :], sal_norm[:, Y // 2, :]),
        ("axial", img_np[:, :, Z // 2], sal_norm[:, :, Z // 2]),
    ]
    for col_offset, (plane, img_slice, sal_slice) in enumerate(slices):
        ax_img = axes[row, col_offset * 2]
        ax_img.imshow(img_slice.T, cmap="gray", origin="lower")
        ax_img.set_title(f"{name}\n{plane} (raw)")
        ax_img.axis("off")

        ax_overlay = axes[row, col_offset * 2 + 1]
        ax_overlay.imshow(img_slice.T, cmap="gray", origin="lower")
        ax_overlay.imshow(sal_slice.T, cmap="hot", alpha=0.55, origin="lower")
        ax_overlay.set_title(f"{plane} + SmoothGrad\n(true={label}, p={prob:.3f})")
        ax_overlay.axis("off")

plt.tight_layout()
out_path = "/tmp/claude-0/-root/8e19c03d-842d-49c8-ac10-4c5f17622188/scratchpad/task5_smoothgrad.png"
plt.savefig(out_path, dpi=110)
print(f"saved to {out_path}")
