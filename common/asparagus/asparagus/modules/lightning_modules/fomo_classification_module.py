"""ClassificationModule variant that also reports test-set AUROC (the FOMO26
competition's actual scoring metric for classification tasks -- Task 1
Infarct, Task 5 Polymicrogyria). Upstream `ClassificationModule.
configure_test_metrics()` only computes Precision/Recall on the test split
(clsreg_module.py) -- AUROC is computed for train/val but never for test.

Mirrors the EXISTING convention already used for train/val metrics in
clsreg_module.py's `training_step`/`validation_step`: raw logits are passed
directly to torchmetrics (`self.train_metrics.update(pred, y)`, no manual
softmax/argmax -- torchmetrics' Multiclass* metrics handle that internally).
The only upstream bug this fixes is that `on_test_batch_end` collapsed
predictions to a hard argmax class *before* they ever reached
`self.predictions`, which is fine for Precision/Recall but throws away the
per-class score AUROC needs -- so this override keeps the full logit vector
instead.

Per-file results store a single `prob_positive` scalar (softmax(logits)[1]),
not the full 2-class vector -- a fold's 2-3 held-out subjects can't support
their own AUROC, so every subject's positive-class probability gets pooled
across all 10 folds before computing ONE overall AUROC (see
aggregate_stratified10.py); storing the full vector invites picking the
wrong column downstream, so only the unambiguous scalar is persisted.
Also records which epoch/val-loss the "best" checkpoint (used for this test
pass) actually came from, for post-hoc fold-stability inspection.
"""

import logging
import os

import torch
from asparagus.modules.lightning_modules.clsreg_module import ClassificationModule
from gardening_tools.functional.paths.write import save_json
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAUROC, MulticlassPrecision, MulticlassRecall


class FomoClassificationModule(ClassificationModule):
    def configure_test_metrics(self):
        return MetricCollection(
            {
                "AUROC": MulticlassAUROC(num_classes=self.num_classes, average=None),
                "Precision": MulticlassPrecision(num_classes=self.num_classes, average=None),
                "Recall": MulticlassRecall(num_classes=self.num_classes, average=None),
            }
        )

    def _best_checkpoint_info(self):
        """Reads back (best_epoch, best_val_loss) for whichever ModelCheckpoint
        callback monitors "val/loss" (finetune_cls.py's `best_ckpt_callback` --
        matched by monitor name, not list position, since two ModelCheckpoint
        callbacks are registered and Lightning's `trainer.checkpoint_callback`
        singular-property selection order isn't something to rely on)."""
        best_cb = next(
            (cb for cb in self.trainer.checkpoint_callbacks if getattr(cb, "monitor", None) == "val/loss"),
            None,
        )
        if best_cb is None or not best_cb.best_model_path:
            return None, None
        best_val_loss = float(best_cb.best_model_score) if best_cb.best_model_score is not None else None
        try:
            ckpt = torch.load(best_cb.best_model_path, map_location="cpu", weights_only=False)
            best_epoch = ckpt.get("epoch")
        except Exception:
            best_epoch = None
        return best_epoch, best_val_loss

    def on_test_epoch_start(self):
        self._best_epoch, self._best_val_loss = self._best_checkpoint_info()
        return super().on_test_epoch_start()

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        prediction = outputs.argmax(1).long()
        label = batch["CLSREG_label"]
        prob_positive = torch.softmax(outputs, dim=1)[0, 1].item()
        self.results[batch["file_path"]] = {
            "prediction": prediction.item(),
            "label": label.item(),
            "prob_positive": prob_positive,
            "best_epoch": self._best_epoch,
            "best_val_loss": self._best_val_loss,
        }
        self.predictions.append(outputs.squeeze(0))  # full per-class logit vector, needed for AUROC metric
        self.labels.append(label)

    def on_test_epoch_end(self):
        preds = torch.stack(self.predictions)
        labels = torch.cat(self.labels)
        avg_results = self.test_metrics(preds, labels)
        avg_results = {key: value.cpu().numpy().tolist() for key, value in avg_results.items()}
        self.results["metrics"] = avg_results
        os.makedirs(os.path.split(self.test_output_path)[0], exist_ok=True)
        save_json(self.results, self.test_output_path)
        logging.info(f"Aggregated test results for {len(self.results)} files: {avg_results}")
