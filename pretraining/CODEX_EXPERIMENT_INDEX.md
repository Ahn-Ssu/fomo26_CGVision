# FOMO26 Codex Experiment Index

Last updated: 2026-08-14 (Task 1 augmentation 30-fold CV stopped at user request: 24/30 folds complete)

This is the canonical index for experiments and audits started or materially
revised by Codex. Experiment IDs never change. Update the existing row instead
of creating a new ID when a job is retried without changing its scientific
question. Create a new ID when the input distribution, split, model-selection
rule, model architecture, or principal hypothesis changes.

## Status vocabulary

- `PLANNED`: design recorded, not launched
- `RUNNING`: one or more jobs active or queued
- `VERIFYING`: training finished; coverage/metrics checks pending
- `COMPLETED`: required jobs and aggregation completed
- `SUPERSEDED`: retained for provenance but replaced by another indexed run
- `BLOCKED`: cannot progress without an external change or decision

## Canonical three-layer contract

Every experiment must be audited as three independent contracts. Matching
YAML/configuration is not evidence that a contract passes. A result is not
considered final until all applicable layers are `PASS`.

Contract status vocabulary:

- `PASS`: executable, artifact-backed comparison completed
- `PARTIAL`: some paths/components verified, at least one required path remains
- `FAIL`: a numerical or behavioral mismatch was demonstrated
- `UNVERIFIED`: no direct executable comparison yet
- `N/A`: demonstrably not applicable, with the reason recorded

### Layer A — Data contract

`raw file -> resampling -> crop -> normalization -> padding -> model-input tensor`

Required canonical sanity test:

1. Select and record one fixed representative case (and edge cases when
   modality count, missing foreground, anisotropy, or oversized crops matter).
2. Pass the same case independently through training, validation, local-test,
   and submission-container paths.
3. Capture the tensor at the immediate model-input boundary, not at an earlier
   preprocessing stage.
4. Compare shape, dtype, channel order, affine/orientation-derived geometry,
   finite values, range, per-channel statistics, foreground/background masks,
   and tensor values.
5. Prefer byte equality when paths use the same operations and serialization.
   Otherwise report `max_abs_diff`, `mean_abs_diff`, RMSE and explicit
   tolerances; a qualitative visual match or identical YAML is insufficient.

Required evidence: case ID, source paths, executable/script, tensor-capture
point, comparison metrics, software/container identity, and output artifact.

### Layer B — Model contract

`model input -> stem/encoder -> requested stages -> PEFT/decoder -> pooling,
fusion, or task head`

Required audit:

1. Record loaded checkpoint and exact state-dict key/shape coverage, including
   missing and unexpected keys.
2. Enumerate parameters by component with `requires_grad`, train/eval state,
   parameter counts and, where relevant, gradient-presence/gradient-norm checks.
3. Record the active teacher/PEFT context and demonstrate that changing context
   activates the intended modules only.
4. Prove weight sharing or non-sharing using module/parameter object identity
   and storage pointers, not architecture names.
5. Capture requested stage tensors with hooks and verify module path, shape,
   stride/spatial scale, channel count and their actual downstream consumer.
6. Audit normalization/stateful layers in train and eval modes; verify that
   evaluation does not update running state and that repeated eval forwards are
   deterministic within an explicit tolerance.

Task-specific minimums:

- Task 1: modality routing, encoder sharing, requested stage features,
  pooling/fusion/classifier, frozen weights and PEFT context.
- Task 2/segmentation: stem, encoder, Convpass, decoder and segmentation head
  parameter-activation/gradient-flow audit.

### Layer C — Evaluation contract

`validation sampling -> checkpoint monitor -> inference -> postprocessing ->
reverse transform -> metric`

Required audit:

1. Enumerate the exact validation and held-out subjects and prove sampler
   cardinality, order/replacement behavior and number of evaluations per case.
2. Record checkpoint monitor, aggregation unit, mode, tie behavior, selected
   epoch/step and the restored checkpoint identity/hash.
3. Compare local and container inference coverage, including sliding-window or
   full-volume settings, overlap, TTA and precision.
4. Compare logits/probabilities or segmentation maps before postprocessing,
   then each postprocessing and reverse-transform stage separately.
5. Run the same metric implementation or demonstrate numerical equivalence,
   including class definition, empty-case behavior, thresholding and averaging.
6. For cross-validation, require exact out-of-fold subject coverage, no
   duplicates/leakage and uncertainty estimation on the pooled predictions.

### Contract verification matrix

| Experiment ID | Layer A | Layer B | Layer C | Current blocking evidence |
|---|---|---|---|---|
| `CX-MK4-CLS1-001` | `PARTIAL` | `UNVERIFIED` | `PARTIAL` | A: train/val/local tensor checks and raw-to-training reproduction passed, but final container capture remains. B: prior code review exists but the formal parameter/context/stage audit has not run. C: all ten OOF folds completed with exact 21-subject coverage; deterministic validation and checkpoint restore passed, but final local-vs-container prediction/postprocessing/metric alignment remains. |

## Experiment registry

Execution update: the 5-epoch mild-spatial sanity completed including
best-checkpoint restore and held-out inference. The original automatic-launch
watcher did not persist, so Codex directly launched the 30-fold orchestrator.
The first job, minimal_fold0, is active; all three augmentation arms are
therefore RUNNING.

| ID | Server | Task | Scientific question / configuration | Status | Completed actions | Current evidence | Next action | Artifacts |
|---|---|---|---|---|---|---|---|---|
| `CX-MK4-CLS1-001` | mk4 | Task 1 infarct classification | Corrected no-double-normalization baseline; ver5 step 280k, pretrained, learnable stem, base head, early fusion, LR 1e-4, 25 epochs, stratified 10-fold | `VERIFYING` | Passed `transforms.normalize=false` into train/val/test transforms; removed replacement sampling from validation; fixed validation to two deterministic subjects in one batch; fixed per-fold seeds; added held-out OOF coverage checks and stratified bootstrap aggregation; completed one-epoch end-to-end smoke test; completed all 10 folds and OOF aggregation | 21 unique held-out subjects (13 positive, 8 negative), folds 0-9 all present. Pooled AUROC **0.576923**, stratified bootstrap 95% CI **[0.322115, 0.817308]** (100,000 resamples). This is below the previously reported, distribution-confounded 0.707 baseline. | Complete Layer A container capture, Layer B parameter/stage audit, and Layer C local-container prediction/metric comparison before marking final | `/root/asparagus_orchestrate_task1_no_norm_corrected.py`; `/root/task1_no_norm_corrected/aggregate_corrected.py`; `/root/task1_no_norm_corrected/output/{predictions.csv,auroc_summary.csv}` |
| `CX-MK4-CLS1-002` | mk4 | Task 1 infarct classification | Augmentation control: mirror-only, no spatial deformation/rotation/scale and no GPU intensity transforms; same corrected baseline and LR 1e-4 | `FOLD-COMPLETE; AGGREGATION PENDING` | Dedicated mirror-only CPU transform; all 10 folds completed. | Tensor preset import and fixed-shape finite-output test passed; all fold logs report optimizer LR 0.0001; no error signatures. | Run OOF coverage and pooled-AUROC aggregation after the remaining arms complete. | `/root/asparagus_orchestrate_task1_augmentation_controls.py`; root `task1_aug_minimal` |
| `CX-MK4-CLS1-003` | mk4 | Task 1 infarct classification | Augmentation treatment: mirror + mild spatial only: rotation P=0.2, per-axis P=0.3, ±15°; scale P=0.2, 0.9–1.1; elastic off; GPU intensity transforms off; LR 1e-4 | `STOPPED (8/10)` | Dedicated mild spatial CPU transform; sanity passed; 8 of 10 full-CV folds completed. The two GPU-unallocated pending folds were stopped at user request. | Fixed model-input shape and finite outputs over 20 calls; every started fold reports optimizer LR 0.0001; no error signatures. | Resume or relaunch folds 2 and 4 before pooled OOF aggregation. | `/root/asparagus_orchestrate_task1_augmentation_controls.py`; root `task1_aug_mild_spatial` |
| `CX-MK4-CLS1-004` | mk4 | Task 1 infarct classification | Augmentation treatment: mirror + mild intensity only. Blur/bias/gamma/ghosting/ringing/multiplicative-noise/additive-noise each channel P=0.05; gamma 0.9–1.1; low-resolution each channel P=0.2, zoom 0.75–1.0; no CPU spatial transform beyond mirror; LR 1e-4 | `STOPPED (6/10)` | Added independent per-channel gamma wrapper and mild GPU-intensity preset; 6 of 10 full-CV folds completed. The four GPU-unallocated pending folds were stopped at user request. | Tensor shape/finiteness passed; all ten launched logs report optimizer LR 0.0001; no error signatures before stop. | Resume or relaunch folds 6-9 before pooled OOF aggregation. | `/root/asparagus_orchestrate_task1_augmentation_controls.py`; root `task1_aug_mild_intensity` |

## Audit and corrective-action registry

| ID | Server/scope | Area | Finding | Action completed | Verification | Remaining action |
|---|---|---|---|---|---|---|
| `CX-AUDIT-001` | mk4 / CLS-REG DataModule | Validation sampling | Validation inherited the training replacement sampler (`num_samples=999999`), so two validation subjects were repeatedly sampled and `val/loss` did not represent one deterministic pass | Added separate opt-in `use_random_val_datasampler=false`; training random sampler left unchanged | Confirmed `SequentialSampler`, 2 subjects, 1 validation batch | Port the same fix to server copies that diverged from mk4 |
| `CX-AUDIT-002` | mk4 / `finetune_cls.py` | Classification normalization | `transforms.normalize` existed in Hydra but was not passed into classification train/val/test transform constructors; `false` therefore silently behaved as default `true` | Passed the flag explicitly to all three classification transforms | Stored `.pt` and transform output were byte-identical with `false`; corrected CV configs record `normalize: false` | Apply equivalent patch to other servers/repository copies; separately fix `train_cls.py` test transform |
| `CX-AUDIT-003` | mk4 / Task 1 preprocessing | Raw NIfTI to `.pt` | Needed to distinguish intended foreground normalization from accidental transform-level normalization | Recomputed representative subjects and traced all four modalities; retained intended 1 mm resample, union crop, per-channel foreground normalization and center pad | Two representative regenerated tensors matched stored `.pt` exactly (`max_abs_diff=0`) and all 21 tensors were finite float32 in `[0,1]` | No preprocessing rewrite required; keep transform-level normalization disabled for these tensors |
| `CX-AUDIT-004` | mk4 / Task 1 inference | Submission preprocessing | Original prediction script did not reproduce the iso1mm training preprocessing and did not honor checkpoint normalization consistently | Added four-channel validation, affine/shape checks, 1 mm intensity/mask resampling, background cleanup, union crop, foreground normalization, pad, temporary `.pt`, and checkpoint-driven transform flag | Representative raw inputs reproduced stored training tensors exactly | Container integration/smoke test with final selected checkpoint |
| `CX-AUDIT-005` | mk4 / pretraining ver1-ver5 | Pretraining normalization | Checked whether the downstream double-normalization bug also affected mk4 pretraining | Confirmed raw-preserving `.npz`; exactly one student-side `asparagus_volume_wise_znorm` in Dataset; no `Torch_Normalize` or Asparagus preset in `FOMO26/run_pretrain.py`; verified V-JEPA cache branch also normalizes once | Real cache sample: actual input `[0,1]`; hypothetical second normalization changed it substantially (RMSE about 0.92). All ver1-ver5 configs used `data_source=preprocessed` | Optional precision improvement: feed raw crop to BraTS teacher preprocessing before any asparagus q0.99 clamp; not a reason to discard ver4/ver5 |
| `CX-AUDIT-006` | cross-task / SEG-CLS-REG | Normalization flag propagation | Same class of configuration-wiring defect existed across task entry points | Classification finetune path fixed and directly verified on mk4; segmentation and regression fixes were propagated and resolved on their respective servers per user report; affected prior Task 1 results marked as distribution-confounded and replaced by `CX-MK4-CLS1-001` | Code diff and live config/batch checks completed for mk4 classification. mk5/mk8 resolution is user-confirmed but has not been independently inspected from mk4 | When artifacts are synchronized, record exact patched entry points and one post-transform batch check for mk5 and mk8; no further propagation action currently required |
| `CX-AUDIT-007` | mk4 / checkpoint selection | CV model selection | Fold validation sets can be single-class, making validation AUROC undefined/unstable; random validation sampling also contaminated selection | Select `save_top_k=1` by minimum mean `val/loss` over the two fixed validation subjects; restore that checkpoint for untouched held-out prediction | One-epoch smoke test covered save, restore and held-out inference | Report per-fold best epoch/loss with final OOF aggregation |
| `CX-AUDIT-008` | mk4 / spatial augmentation | Scale-probability propagation | `CPU_seg_train_transforms` declared `p_scale_all_channel=0.2` but failed to pass it to `Torch_Spatial`, which defaulted to scale probability 1.0 | Passed `p_scale_all_channel=p_scale_all_channel` explicitly into the segmentation spatial transform | Direct object inspection: segmentation, default classification, and mild classification presets all expose `p_scale_all_channel=0.2`; gardening-tools forwards it as `p_scale` to its Bernoulli scale branch | Port the one-line segmentation preset fix to any divergent server copy; do not alter classification, which was already correct |
| CX-AUDIT-009 | mk4 / Task 1 augmentation | Learning-rate override | Shared FOMO26 model config defaults finetune LR to 1e-3, so YAML-only assertion is insufficient | Each classification launch explicitly passes model.finetune_lr=1e-4 after model selection; finetune_cls forwards this field to the Lightning optimizer | Resolved configs and all 30 launched fold logs, including mild intensity, print optimizer learning rate 0.0001 | Closed. No corrective rerun is needed for these mk4 jobs. |

## Known external work (not yet Codex-indexed)

These rows are visibility notes only. They do not receive a `CX-*` experiment
ID until Codex is given the exact run configuration/artifacts or materially
revises/launches the experiment.

| Server | Task | Last user-provided state | Codex visibility |
|---|---|---|---|
| mk5 | Tasks 2 and 4 | Running on four 24 GB GPUs; segmentation normalization/config-wiring bug propagated and resolved on mk5 per user report | Resolution recorded as user-confirmed; exact live patch and post-transform batch have not been independently inspected from mk4 |
| mk8 | Task 3 age regression | Running on two RTX 5090 32 GB GPUs; regression normalization/config-wiring bug propagated and resolved on mk8 per user report | Resolution recorded as user-confirmed; exact live patch and post-transform batch have not been independently inspected from mk4 |
| other | Tasks 6 and 7 | Cannot be trained | No active training |

## Required update fields for every future report

Every Codex experiment update will include: stable ID, server, task, exact
hypothesis/config delta, current status, completed corrective actions,
verification evidence, artifact/log paths, and the single next action. Final
reports additionally include subject/sample coverage, checkpoint-selection
rule, primary metric with uncertainty, whether the run supersedes an earlier
result, and explicit Layer A/B/C contract statuses with artifact-backed
evidence. No experiment is labeled final while an applicable contract is
`PARTIAL`, `FAIL`, or `UNVERIFIED`.
