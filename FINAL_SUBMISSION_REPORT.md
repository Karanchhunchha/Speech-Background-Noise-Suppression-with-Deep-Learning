# Final submission report — Project #193

**Project:** Speech Background Noise Suppression with Deep Learning
(MathWorks Excellence in Innovation, Project #193)
**Author:** Karan Chhunchha — HNGU Patan, Department of Computer Science
**Repository:** https://github.com/Karanchhunchha/Speech-Background-Noise-Suppression-with-Deep-Learning
**Verified on:** MATLAB Online, R2026b, 27 November 2026

## What the brief asked for, and what was delivered

| Brief item | Delivered |
|---|---|
| Literature / background study | README "Literature in brief" discusses all eight of the brief's suggested readings (DNS Challenge, RNNoise, FullSubNet, dual-signal transformation LSTM, phase-aware U-Net, PoCoNet, Noisy2Noisy, WaveNet denoising) and the classical roots (Boll 1979, Ephraim–Malah 1984, Taal et al. 2011), and states why the compact-CNN IRM design point was chosen; the MathWorks denoising example was the implementation reference |
| Dataset of noisy/clean speech | VoiceBank-DEMAND (CC BY 4.0), the standard academic benchmark for this task — used instead of the full Microsoft DNS corpus (hundreds of GB); the deviation and its reason are stated in the README. `download_data()` fetches and extracts 824 test pairs + 800 training pairs |
| Deep learning network for noise suppression | Compact fully-convolutional CNN (~24k parameters) estimating an ideal ratio mask from a log-magnitude spectrogram, Deep Learning Toolbox, noisy-phase resynthesis |
| Validation, objective | Segmental SNR, log-spectral distance and STOI over the full 824-file standard test set (speakers p232/p257, unseen in training), against two classical baselines |
| Validation, subjective | `demo_denoise()` writes before/after .wav files and spectrogram figures for the committed samples |
| Single main entry point (hub repository guidelines) | `main()` runs the whole system end to end — tests, demo, evaluation — with `main('Full', true)` adding corpus download and training |

## Verified results (full 824-file test set)

| Method | Segmental SNR (dB) | LSD (dB) | STOI |
|---|---|---|---|
| Noisy input | 1.52 | 14.50 | 0.921 |
| Spectral subtraction (Boll) | 2.08 | 13.60 | 0.922 |
| Wiener, decision-directed | 2.99 | 12.71 | 0.919 |
| **CNN (shipped model)** | **5.88** | **11.37** | 0.912 |

- CNN vs noisy input: **+4.36 dB** segmental SNR, **−3.13 dB** LSD.
- CNN vs strongest baseline (Wiener): +2.89 dB segmental SNR, −1.33 dB LSD.
- STOI of the CNN output (0.912) is 0.009 **below** the noisy input
  (0.921). This is reported, not hidden: aggressive mask-based
  suppression trained with a mask-MSE objective trades a small amount
  of envelope fidelity for large noise-energy removal. It is the
  project's main stated limitation and the first target for future
  work (a perceptually weighted loss).

## Model selection (measured, not assumed)

Two configurations were trained and evaluated on the identical full
test set; the better one ships. (The first configuration's summary
output is kept as `reports/metrics_summary_v1.csv`.)

| Configuration | SegSNR (dB) | LSD (dB) | STOI |
|---|---|---|---|
| 300 utterances, 1,800 patches, 12 epochs | 4.59 | 11.45 | 0.908 |
| **800 utterances, 4,800 patches, 24 epochs (shipped)** | **5.88** | **11.37** | **0.912** |

Shipped training run: 4,800 patches prepared from 800 utterances,
24 epochs in ~55 minutes on CPU (MATLAB Online); the training
progress showed a mini-batch RMSE of ≈ 20 in the last visible
epochs (20–23 of 24; the session timed out during epoch 24 and the
run's opening values were lost with it — earlier training runs of
this pipeline started at ≈ 122). The saved artifact was verified on
disk afterwards: correct patch statistics, and it produced the
evaluation below.
Artifact `models/denoiseNet.mat` (297,335 bytes). `train_model()`
defaults now match this configuration exactly.

## Test suite

`run_all_tests()` in MATLAB Online (R2026b): **25 passed, 0 failed,
0 incomplete.** The suite covers the STFT/ISTFT round trip, all three
metrics, both baselines, the IRM target, the network definition,
inference on a real sample pair (including the short-file padding
branch), the chunk-merge and its overlap guard, the committed data,
and an end-to-end chain.

Final-state re-verification (after an independent code and security
review pass): the suite above ran on the final code; a fresh
`download_data()` re-fetched the corpus with both SHA-256 checks
passing (824 test + 800 training pairs extracted); and a 10-file
spot evaluation on the final code reproduced the committed per-file
values to the fourth decimal. CNN inference measured 0.173 s for a
1.74 s clip on a MATLAB Online CPU — a real-time factor of ≈ 0.10,
about 10× faster than real time.

## A verification catch worth reporting

The first STOI implementation used a 512-sample analysis window
where the reference algorithm (Taal et al.) uses 256 samples
zero-padded to a 512-point FFT. The longer window over-smooths the
band envelopes and inflated scores badly (noisy p232_001 read 0.9997
instead of 0.898). Cross-checking against the reference
implementation (pystoi) exposed it; the corrected implementation
matches the reference within 0.0005 on all 8 committed sample pairs,
and the full-test noisy-input STOI (0.921) now agrees with the value
reported in the VoiceBank-DEMAND literature. The episode is
documented in the `stoiScore` header.

## Reproduction

```matlab
ok = run_all_tests()          % 25/25 on sample data, no downloads
demo_denoise()                % listening demo on committed samples
download_data()               % full corpus (test 824 + train 800 pairs)
train_model()                 % shipped configuration, ~55 min CPU
summary = evaluate_models()   % rewrites reports/ from the full test set
```

## Known limitations

- STOI slightly below the noisy input, as discussed above.
- Noisy-phase resynthesis (standard for mask-based systems).
- Trained on 800 of the corpus's 11,572 training pairs, CPU-bounded;
  the pipeline exposes the knobs to scale up.
- PESQ is not reported: it requires the Audio Toolbox / a licensed
  implementation, and this project is deliberately toolbox-free
  outside the Deep Learning Toolbox.
- Training uses the `trainNetwork` API (current and verified on
  R2026b); MathWorks now recommends `trainnet` for new code, so a
  future MATLAB release may require that migration.

---

## Addendum — submission-hardening round (2026-11-27, later the same day)

The sections above describe the repository as first submitted.
The same evening, a hardening round was run against the hub's
Repository Guidelines and the project brief. What changed, with
the before/after measured on the full 824-file test set:

| CNN | SegSNR (dB) | LSD (dB) | STOI | SI-SDR (dB) |
|---|---|---|---|---|
| v2, 800 utts / 24 epochs, raw mask | 5.88 | 11.37 | 0.912 | — |
| R1, 1,600 utts / 16 epochs, floor 0.10 / exp 0.7 | 5.63 | 11.90 | 0.920 | 14.20 |

- **Model selection is now evidence-based.** R1 was trained with
  two speakers held out and beat v2 on every configuration and
  metric on 145 held-out validation files (at the matched raw-mask
  setting: SegSNR 4.16 vs 3.61). The shipped operating point
  (mask floor 0.10, exponent 0.7) was chosen on that validation —
  before the single test evaluation — and trades 0.25 dB of test
  SegSNR for +0.0075 STOI: the below-noisy STOI gap shrank from
  −0.009 to −0.001. Full log: `reports/ablation.md`; v2 frozen in
  `reports/baseline_v1.csv`.
- **Evaluation grew up:** SI-SDR added; per-SNR breakdown
  (`reports/metrics_by_snr.csv`, mixture SNR measured per file
  from its clean reference); a PESQ route exists offline
  (`tools/python_metrics.py`) — PESQ numbers are reported only
  where that script has actually run, with their N.
- **Verification grew up:** test suite 25 → 26, adding a
  cross-check of the hand-written STFT against the Signal
  Processing Toolbox `stft` (passes; the toolbox's centred padding
  convention differs, so the test compares transform math on a
  pre-padded signal). `verify_submission()` gives a reviewer a
  one-command PASS/FAIL audit, including reproducing the committed
  metrics on 8 files (worst deviation 0.00 of tolerance).
- **Real-time measured:** simulated streaming in 256 ms chunks —
  CNN mean 18.1 ms, worst 83.3 ms per chunk (real-time factor
  ≈ 0.07), with its ≈ 1.02 s segment look-ahead disclosed
  (`demo_realtime.m`, `reports/realtime_results.csv`).
- **Required products, machine-generated:**
  `reports/required_products.txt` — MATLAB + Deep Learning
  Toolbox only. The Audio Toolbox was investigated thoroughly and
  is not usable in the verification environment (its functions
  fail their licence checkout there); nothing in the pipeline
  depends on it.
- Two limitation bullets above are superseded by this addendum:
  the model is now trained on 1,600 utterances, and the STOI gap
  is 0.001, not 0.009. The `trainNetwork` note stands.
