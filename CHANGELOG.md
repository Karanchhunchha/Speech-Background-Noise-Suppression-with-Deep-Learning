# Changelog

All notable changes to this project are recorded here. Versions
follow the submission history; dates are IST.

## [Unreleased] — advanced upgrade round (2026-11-28)

### Added
- **Log-MMSE baseline** (`src/baselines/logMmseEnhance.m`) —
  Ephraim–Malah log-spectral amplitude estimator, the canonical
  classical reference, wired into `evaluate_models`,
  `demo_realtime` and the Python metrics tool.
- **`buildfile.m`** — MATLAB build tool plan: `check` (code
  issues), `test` (unit tests), `verify` (full submission
  self-check), `clean`.
- **v3 complex-mask model code** (`src/dl/buildDenoiseNetV3.m`,
  `modelLossCRM.m`, `loadDenoiseModel.m`, `train_model_v3.m`) —
  residual dilated CNN predicting a complex ratio mask, custom
  `dlnetwork` training loop with per-epoch checkpoints.
  Candidate only: it failed held-out validation (see
  `reports/ablation.md`) and is **not** the shipped model.
- `tools/validate_model.m` — speaker-held-out ship-decision
  harness used for every model candidate.
- `tools/recalibrate_bn_v3.m` — recomputes batch-norm trailing
  averages for a trained candidate (repairs the v3 run below).
- `docs/ARCHITECTURE.md` — system architecture and design notes.
- Tests: 30 total (was 26), including a Signal Processing
  Toolbox STFT cross-check and v3 network/loss/smoke tests.

### Fixed
- **v3 training loop dropped the batch-norm state** returned by
  `forward` inside the loss, so the saved candidate kept the
  initial trailing averages and `predict` exploded at inference
  (validation SegSNR −9.23 dB despite a healthy training loss).
  State is now returned by `modelLossCRM` and assigned in
  `train_model_v3`. Full account in `reports/ablation.md`.

## [1.0.0] — 2026-11-28

- Hardened final state merged to `main` (PR #1, `dd26525b`):
  R1 ideal-ratio-mask CNN shipped with mask floor 0.10 /
  exponent 0.7 (validation-selected), full 824-file test
  evaluation, offline PESQ subset, per-SNR breakdown, real-time
  demo, `verify_submission` all-checks-passed, 26/26 tests.

## [0.1.0] — 2026-11-27

- Initial build and hub submission: toolbox-free STFT/ISTFT and
  metrics (STOI verified against the reference implementation),
  spectral-subtraction and Wiener baselines, CNN IRM model,
  VoiceBank-DEMAND data pipeline with pinned revision and
  SHA-256 verification.
