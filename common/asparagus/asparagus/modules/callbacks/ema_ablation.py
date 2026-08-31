import csv
import os

import torch
import torch.nn.functional as F
import lightning as pl


def load_subjects(test_paths):
    """Load a fold's held-out subjects from their preprocessed .pt paths --
    same loading pattern used throughout the task1_arch18 post-hoc aggregate
    scripts (e.g. asparagus_aggregate_task1_champion_ensemble.py)."""
    subjects = []
    for p in test_paths:
        image, label = torch.load(p, map_location="cpu", weights_only=False)
        sub = [part for part in p.split("/") if part.startswith("sub-")][0]
        subjects.append((sub, image, int(label.item())))
    return subjects


class EmaAblationCallback(pl.Callback):
    """Champion-recipe stabilization ablation (2026-08-17), extended
    2026-08-18 to actually persist EMA weights to disk (the original version
    only ever evaluated the EMA shadow transiently and discarded it -- fine
    for the ablation question itself, but it meant there was no way to
    recover a deployable EMA checkpoint after the fact without re-training,
    which is exactly what forced the epoch-20 EMA submission to be a full
    10-fold re-run instead of a reuse of already-trained weights. Not
    repeating that mistake here.

    Maintains EMA shadow copies of the model's trainable parameters (only
    the small PEFT Convpass adapters + head -- backbone is frozen) at
    several decay rates, updated after every optimizer step. Every
    `eval_every_n_steps`:
      1. evaluates raw weights AND each EMA variant against the fold's
         held-out subjects, appending per-subject probabilities to a CSV
         (same as before -- pooled AUROC per (variant, step) computed
         post-hoc across folds).
      2. saves a small bundle per beta to `ckpt_save_dir` (only the
         trainable-parameter subset -- a few MB, not a full model
         checkpoint) containing raw_state_dict + ema_state_dict +
         global_step + beta. The frozen backbone (~99.98% of the model's
         parameters) never changes during finetuning, so it is NOT
         duplicated here -- reconstruct a full model at any saved step by
         taking the frozen backbone from any of that fold's regular
         (last_ckpt_callback-saved) full checkpoints and overwriting the
         trainable subset with either raw_state_dict or ema_state_dict from
         this bundle.

    Goal is comparing within-run checkpoint variance (not between-run seed
    variance, which the champion 3-seed sweep already showed is small) --
    does EMA reduce late-training plateau fluctuation relative to raw
    weights, even at the cost of a lower peak.
    """

    def __init__(self, betas, eval_every_n_steps, subjects, out_csv_path, ckpt_save_dir=None):
        super().__init__()
        self.betas = list(betas)
        self.eval_every_n_steps = eval_every_n_steps
        self.subjects = subjects
        self.out_csv_path = out_csv_path
        self.ckpt_save_dir = ckpt_save_dir
        self.shadows = None  # {beta: {param_name: tensor}}
        self._header_written = os.path.exists(out_csv_path)
        if self.ckpt_save_dir:
            os.makedirs(self.ckpt_save_dir, exist_ok=True)

    def _variant_name(self, beta):
        return "ema" + str(beta).split(".")[1]

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        model = pl_module.model
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

        if self.shadows is None:
            self.shadows = {beta: {n: p.detach().clone() for n, p in trainable} for beta in self.betas}
        else:
            with torch.no_grad():
                for beta, shadow in self.shadows.items():
                    for n, p in trainable:
                        shadow[n].mul_(beta).add_(p.detach(), alpha=1 - beta)

        step = trainer.global_step
        if step > 0 and step % self.eval_every_n_steps == 0:
            self._evaluate_all(model, trainable, step)

    @torch.no_grad()
    def _evaluate_all(self, model, trainable, step):
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device

        rows = self._evaluate_variant(model, "raw", step, device)

        raw_backup = {n: p.detach().clone() for n, p in trainable}
        if self.ckpt_save_dir:
            for beta in self.betas:
                self._save_shadow_bundle(raw_backup, beta, step)

        for beta in self.betas:
            for n, p in trainable:
                p.copy_(self.shadows[beta][n])
            rows += self._evaluate_variant(model, self._variant_name(beta), step, device)

        for n, p in trainable:
            p.copy_(raw_backup[n])

        if was_training:
            model.train()

        self._write_rows(rows)

    def _save_shadow_bundle(self, raw_backup, beta, step):
        bundle = {
            "raw_state_dict": {n: t.cpu() for n, t in raw_backup.items()},
            "ema_state_dict": {n: t.cpu() for n, t in self.shadows[beta].items()},
            "global_step": step,
            "beta": beta,
        }
        path = os.path.join(self.ckpt_save_dir, f"{self._variant_name(beta)}_step{step:06d}.pt")
        torch.save(bundle, path)

    def _evaluate_variant(self, model, variant, step, device):
        rows = []
        for sub_id, image, label in self.subjects:
            x = image.unsqueeze(0).to(device)
            logits = model(x)
            prob = F.softmax(logits, dim=1)[0, 1].item()
            rows.append({"variant": variant, "step": step, "subject_id": sub_id, "label": label, "prob_positive": prob})
        return rows

    def _write_rows(self, rows):
        os.makedirs(os.path.dirname(self.out_csv_path), exist_ok=True)
        with open(self.out_csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["variant", "step", "subject_id", "label", "prob_positive"])
            if not self._header_written:
                w.writeheader()
                self._header_written = True
            w.writerows(rows)
