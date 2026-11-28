# Architecture

How this project is put together, and why. For results see
`reports/EVALUATION.md`; for the model-selection protocol see
`reports/ablation.md`.

## Pipeline

```
download_data        VoiceBank-DEMAND (pinned revision, SHA-256
                     verified) -> data/voicebank/{clean,noisy}_{train,test}
prepare_training_data  STFT -> log-magnitude patches + IRM targets
                     -> data/training_patches.mat      (model v1)
train_model          trainNetwork, regression head     (model v1)
train_model_v3       custom dlnetwork loop, on-the-fly random crops,
                     per-epoch checkpoints             (model v3)
evaluate_models      all methods on the 824-file test set ->
                     reports/metrics_{per_file,summary,by_snr}.csv,
                     reports/EVALUATION.md
demo_denoise         before/after audio + spectrograms for one file
demo_realtime        simulated streaming timings per method
verify_submission    the self-check gate (products, tests, metric
                     reproduction, README consistency, model loads)
buildtool            check + test + verify in one command (buildfile.m)
```

## Signal processing core (`src/audio`, `src/metrics`)

Everything is hand-written base MATLAB — no toolbox dependency —
so the project runs on a stock MATLAB + Deep Learning Toolbox
install:

- `stft` / `istft`: 512-sample Hann window, 256 hop, overlap-add
  with window-envelope normalisation. Cross-checked against the
  Signal Processing Toolbox `stft` in
  `tests/test_ToolboxCrossCheck.m`.
- Metrics: segmental SNR, log-spectral distance, SI-SDR, and STOI.
  The STOI implementation follows Taal et al. (2011) exactly
  (256-sample analysis window zero-padded to 512, one-third-octave
  bands snapped to DFT bins, time-domain silence removal) and was
  verified against the reference `pystoi` implementation to within
  ±0.0005 — an earlier draft that deviated from the reference
  windowing inflated scores to 0.9997 and was caught and fixed by
  exactly this cross-check.

## Classical baselines (`src/baselines`)

All three share the same STFT framing and the same noise estimate
(`noiseEstimate`: per-bin 10th-percentile magnitude over time,
frequency-smoothed), so the comparison isolates the estimator:

- **Spectral subtraction** (Boll 1979): power subtraction,
  α = 2 over-subtraction, β = 0.01 spectral floor.
- **MMSE-STSA / Wiener** (Ephraim–Malah 1984): decision-directed
  a priori SNR (smoothing 0.98), gain ξ/(1+ξ).
- **Log-MMSE** (Ephraim–Malah 1985): gain
  (ξ/(1+ξ))·exp(½·E₁(γξ/(1+ξ))) with `expint` — the canonical
  strong classical reference (Loizou benchmarks); less aggressive
  at low SNR, so less speech distortion and musical noise.

## Learned models (`src/dl`)

- **v1 — IRM CNN (shipped).** Five 3×3 convolutions (~24k
  parameters) mapping a standardised log-magnitude segment
  (257×64) to an ideal ratio mask; noisy phase reused at
  synthesis. Trained with `trainNetwork` on precomputed patches
  (`prepare_training_data`). Inference applies the mask in
  overlapping segments with a validation-chosen operating point
  (floor 0.10, exponent 0.7 — see `reports/ablation.md`).
- **v3 — complex-mask residual CNN (candidate).** Four residual
  blocks (48 channels) with time-dilated convolutions (1/2/4/8),
  ~180k parameters, predicting a **complex** ratio mask
  (2·tanh-bounded) from a 3-channel input (noisy real, noisy
  imaginary, standardised log-magnitude) — phase is corrected,
  not reused. Trained on the full training corpus with a custom
  `dlnetwork` loop (`train_model_v3`): one fresh random crop per
  utterance per epoch, masked-spectrum loss (complex MSE +
  0.5 × magnitude MSE, `modelLossCRM`), Adam with gradient
  clipping, and a resumable checkpoint after every epoch.

`loadDenoiseModel` tags every artifact with a `kind` (`irm` /
`crm`); `denoiseWithNet` dispatches on it, so both generations
run through the same evaluation, demo and verification code.

## Model-selection protocol

No configuration is chosen on the test set. Candidates are
trained with speakers **p236 + p254 held out**, ranked on those
speakers' 145 training-corpus files with `tools/validate_model.m`,
and the winner is evaluated **exactly once** on the full test
set. Operating points, ship decisions and their evidence are
recorded in `reports/ablation.md`, including the cases where a
candidate lost.

## Optional tools (`tools/`)

Not required by the core pipeline: `pesq_native` (documents why
in-MATLAB PESQ cannot exist — MATLAB ships no PESQ function;
verified), `python_metrics.py` (offline PESQ on exported WAVs),
`ablate_mask` (inference-time operating-point grid on validation),
`make_results_chart`, `make_listening_test` /
`analyze_listening_test` (blinded listening test),
`update_readme_results` (regenerates the README results table
from the summary CSV).
