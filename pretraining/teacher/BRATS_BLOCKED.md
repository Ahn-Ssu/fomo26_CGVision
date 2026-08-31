# BraTS teacher -- RESOLVED (2026-07-15)

**Checkpoint delivered to `/root/teachers/BarTS_Teacher/` on 2026-07-15. Wrapper
implemented, `strict=True` verified, registered in `TEACHER_REGISTRY` as
`"brats"`, passes all smoke tests including real-MRI sanity check. See
`/root/teachers/wrappers/brats_teacher.py` (full docstring with exact
citations) and `/root/FOMO26/CLAUDE_CODE_RUN_REPORTS/02_TEACHER_REPORT.md` for the up-to-date findings. Everything
below this line is the original blocker checklist, kept for history.**

---

# (historical) BraTS teacher -- BLOCKED

**Status (2026-07-13)**: `02_setup_teachers_v2.md` assumed a trained checkpoint already
exists at:
```
/root/data/for_nnUNet/nnUNet_results/Dataset104_BraTS_1ch_binary/
    nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth
```
This was checked directly on this box (`/root/data` scanned) -- **it does not exist here.**
Asked the user directly; confirmed: *"다른 서버에 존재하는데, 추출해서 추후에 전달할 예정"*
(exists on a different server, will be extracted and provided later).

Everything below is prepared so that once the checkpoint arrives, finishing this teacher
is a bounded, mechanical task -- not a fresh investigation.

## What's already in place
- `/root/teachers/wrappers/brats_teacher.py` -- `BraTSTeacher(BaseTeacher)` skeleton,
  correct constructor contract (`checkpoint_path` = nnUNet training-output directory
  containing `checkpoint_best.pth` + `plans.json` + `dataset.json`), but `_build_model`
  raises `NotImplementedError` -- nothing has been executed against real weights.
- **NOT** registered in `/root/teachers/wrappers/registry.py`'s `TEACHER_REGISTRY` --
  intentional, per that file's own docstring convention for blocked entries.

## Checklist to run once the checkpoint is delivered

1. Copy/extract the delivered checkpoint directory to
   `/root/teachers/checkpoints/brats/` preserving the nnUNet layout
   (`.../fold_0/checkpoint_best.pth`, and a `plans.json` + `dataset.json` either
   alongside it or one level up -- confirm the actual layout you receive, nnUNet's
   convention has varied across versions).
2. `source /root/teachers/venv/bin/activate && pip install nnunetv2`, then
   immediately verify the environment pitfall didn't recur:
   `python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
   must still print `2.6.0+cu124 True`. If not, see the recovery steps used for the
   other three teachers (pin `torch==2.6.0 torchvision==0.21.0 --index-url
   https://download.pytorch.org/whl/cu124`, remove stray `nvidia-*-cu13` packages).
3. Load `plans.json` + `dataset.json`, reconstruct the network via nnunetv2's actual
   plans-to-network utility (find and confirm the real function/class name and
   signature in the installed nnunetv2 version -- do not trust any guessed API call).
   This should be a **ResEnc UNet** per PRE_ANALYSIS.md's Sec 1.1/5, i.e. structurally
   the same family as our own student and as the resenc_unet wrapper conventions
   already established in this repo (`asparagus/modules/networks/resenc_unet.py`,
   `gardening_tools/modules/networks/components/{decoders,encoders}.py` --
   `decoder.decoder_convN`, `encoder.stages[i]` naming, see `/root/FOMO26/CLAUDE_CODE_RUN_REPORTS/01_ASPARAGUS_ANALYSIS.md`).
4. `model.load_state_dict(checkpoint['network_weights'], strict=True)` (nnUNet
   checkpoints store weights under a `network_weights` key, not `state_dict` --
   confirm this against the actual checkpoint dict's top-level keys first).
5. Dump `named_modules()`, confirm decoder stage submodule names, register forward
   hooks (or find a native multi-feature return path if one exists) exactly as done
   for VesselFM (`/root/teachers/wrappers/vesselfm_teacher.py`) and Anatomix
   (`/root/teachers/wrappers/anatomix_teacher.py`).
6. **Normalization -- do not assume.** Read `plans.json`'s
   `"foreground_intensity_properties_per_channel"` and the trainer's actual
   normalization scheme (nnUNet v2 typically applies per-channel `ZScoreNormalization`
   using dataset-wide foreground statistics baked into plans.json, OR per-case
   z-score depending on how the dataset fingerprint was configured -- these are
   NOT the same thing, and only one of them is close to asparagus's own
   `volume_wise_znorm`, which is *per-case*, not dataset-wide). Run the exact same
   empirical Path A vs Path B check used for VesselFM/Anatomix/FastSurfer, using
   `/root/data/FOMO-MRI/fomo-60k/sub_11043/ses_1/t1.nii.gz` as the real test volume
   and asparagus's real `volume_wise_znorm` (verbatim from
   `/root/asparagus_preprocessing/asparagus_preprocessing/utils/normalize.py`, not
   reimplemented from memory). Only set `norm_type` in the wrapper once this is
   actually measured.
7. Implement `_register_hooks`/`preprocess`/`feature_specs`/`norm_type`/
   `input_requirements` for real, remove the `NotImplementedError`s.
8. Run the same 5-point smoke test as the other teachers (frozen params, eval mode,
   deterministic output, forward timing + peak GPU memory on a real 128^3 patch,
   strict=True confirmation) -- see `/root/teachers/tests/verify_vesselfm.py` for
   the pattern to copy.
9. Register in `TEACHER_REGISTRY` in `registry.py`.
10. Re-run `/root/teachers/scripts/verify_teachers.py` (once it exists) with all
    four teachers to confirm joint GPU memory footprint is still acceptable.

## Known reference numbers to cross-check against (already validated in PRE_ANALYSIS.md,
not re-verified here since it predates this session and no checkpoint is available to
re-run it against)

5-fold CV whole-tumor Dice/NSD by modality, quoted from PRE_ANALYSIS.md Sec 5.1 -- treat
as an unverified claim from the planning doc until re-derived from the actual delivered
checkpoint + its own eval logs:

| Modality | Dice | NSD |
|---|---|---|
| t2f (FLAIR) | 0.926 | 0.824 |
| t2w (T2) | 0.901 | 0.734 |
| t1c | 0.861 | 0.601 |
| t1n (T1) | 0.856 | 0.575 |
