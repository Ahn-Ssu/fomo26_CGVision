"""RegressionModule variant that also records which epoch/val-loss the "best"
checkpoint (used for this test pass) actually came from -- same diagnostic
FomoClassificationModule adds for Task 1 classification (see that file's
docstring), reused here for Task 3 (Brain Age) so per-fold best_epoch
stability is inspectable post-hoc the same way.
"""

import torch
from asparagus.modules.lightning_modules.clsreg_module import RegressionModule


class FomoRegressionModule(RegressionModule):
    def _best_checkpoint_info(self):
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
        prediction = outputs.squeeze().float()
        label = batch["CLSREG_label"].squeeze().float()
        self.results[batch["file_path"]] = {
            "prediction": prediction.item(),
            "label": label.item(),
            "best_epoch": self._best_epoch,
            "best_val_loss": self._best_val_loss,
        }
        self.predictions.append(prediction)
        self.labels.append(label)
