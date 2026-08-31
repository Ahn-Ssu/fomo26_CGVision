"""Quick sanity check: run a handful of real training steps (student +
episodic teacher + projection heads + distillation loss + backward) on a
small FOMO300K subset, single process, no accelerate/DDP -- for fast
iteration while editing networks/student.py or losses/distillation.py
without waiting for a full accelerate launch.

Usage: python3 /root/FOMO26/examples/sanity_check_train_step.py [--steps N] [--limit N] [--data_source preprocessed|zip]
"""

import argparse
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/root")
from FOMO26.data.fomo300k_dataset import FOMO300KDataset, FOMO300KPreprocessedDataset  # noqa: E402
from FOMO26.losses.distillation import episodic_distillation_loss  # noqa: E402
from FOMO26.networks.projections import build_projection_heads  # noqa: E402
from FOMO26.networks.student import build_student  # noqa: E402
from FOMO26.sampler.episodic import EpisodicTeacherSampler  # noqa: E402
from FOMO26.teacher.registry import get_teacher  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--data_source", type=str, default="preprocessed", choices=["preprocessed", "zip"])
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--patch_size", type=int, default=128)
    p.add_argument("--teachers", type=str, nargs="+", default=["anatomix", "vesselfm", "brats"])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.data_source == "preprocessed":
        dataset = FOMO300KPreprocessedDataset(patch_size=args.patch_size, limit=args.limit, seed=0)
    else:
        dataset = FOMO300KDataset(patch_size=args.patch_size, limit_zips=args.limit, seed=0)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    student = build_student(in_channels=1).to(device)
    teachers = {name: get_teacher(name, device=device) for name in args.teachers}
    teacher_specs = {name: t.feature_specs for name, t in teachers.items()}
    proj_heads = build_projection_heads(student.decoder_stage_channels, student.encoder.out_channels,
                                         teacher_specs).to(device)

    optimizer = torch.optim.AdamW(list(student.parameters()) + list(proj_heads.parameters()), lr=1e-4)
    sampler = EpisodicTeacherSampler(args.teachers, seed=0)

    data_iter = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        images = batch["image"].to(device)
        mask = batch["mask"].to(device)
        teacher_name = sampler.sample()
        teacher = teachers[teacher_name]

        t0 = time.time()
        student_feats = student.forward_with_features(images)
        teacher_feats = teacher.extract_features(images, meta={"mask": mask})
        projected = proj_heads.project(teacher_name, student_feats)
        losses = episodic_distillation_loss(projected, teacher_feats)
        loss = losses["total"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        dt = time.time() - t0

        print(f"step {step+1}/{args.steps} teacher={teacher_name:10s} loss={loss.item():.4f} ({dt*1000:.0f}ms)")

    print("\nsanity_check_train_step: OK -- pipeline runs end-to-end.")
