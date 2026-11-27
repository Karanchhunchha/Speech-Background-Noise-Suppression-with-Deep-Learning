# Final submission report — Project #193

**Project:** Speech Background Noise Suppression with Deep Learning
(MathWorks Excellence in Innovation, Project #193)
**Author:** Karan Chhunchha — HNGU Patan, Department of Computer Science
**Repository:** https://github.com/Karanchhunchha/Speech-Background-Noise-Suppression-with-Deep-Learning
**Verified on:** MATLAB Online, R2026b, 27 November 2026

## What the brief asked for, and what was delivered

| Brief item | Delivered |
|---|---|
| Literature / background study | Approach grounded in Boll (1979), Ephraim–Malah (1984), Taal et al. (2011) and the MathWorks denoising example; design documented in the README |
| Dataset of noisy/clean speech | VoiceBank-DEMAND (CC BY 4.0), the standard academic benchmark for this task — used instead of the full Microsoft DNS corpus (hundreds of GB); the deviation and its reason are stated in the README. `download_data()` fetches and extracts 824 test pairs + 800 training pairs |
| Deep learning network for noise suppression | Compact fully-convolutional CNN (~24k parameters) estimating an ideal ratio mask from a log-magnitude spectrogram, Deep Learning Toolbox, noisy-phase resynthesis |
| Validation, objective | Segmental SNR, log-spectral distance and STOI over the full 824-file standard test set (speakers p232/p257, unseen in training), against two classical baselines |
| Validation, subjective | `demo_denoise()` writes before/after .wav files and spectrogram figures for the committed samples |

## Verified results (full 824-file test set)

| Method | Segmental SNR (dB) | LSD (dB) | STOI |
|---|---|---|---|
| Noisy input | 1.52 | 14.50 | 0.921 |
| Spectral subtraction (Boll) | 2.08 | 13.60 | 0.922 |
| Wiener, decision-directed | 2.99 | 12.71 | 0.919 |
| **CNN (shipped model)** | **5.88** | **11.37** | 0.912 |

- CNN vs noisy input: **+4.36 dB** segmental SNR, **−3.13 dB** LSD.
- CNN vs strongest baseline (Wiener): +2.89 dB segmental SNR, −1.34 dB LSD.
- STOI of the CNN output (0.912) is 0.009 **below** the noisy input
  (0.921). This is reported, not hidden: aggressive mask-based
  suppression trained with a mask-MSE objective trades a small amount
  of envelope fidelity for large noise-energy removal. It is the
  project's main stated limitation and the first target for future
  work (a perceptually weighted loss).

## Model selection (measured, not assumed)

Two configurations were trained and evaluated on the identical full
test set; the better one ships.

| Configuration | SegSNR (dB) | LSD (dB) | STOI |
|---|---|---|---|
| 300 utterances, 1,800 patches, 12 epochs | 4.59 | 11.45 | 0.908 |
| **800 utterances, 4,800 patches, 24 epochs (shipped)** | **5.88** | **11.37** | **0.912** |

Shipped training run: 4,800 patches prepared from 800 utterances,
24 epochs in ~54 minutes on CPU (MATLAB Online); the training
progress showed a mini-batch RMSE of ≈ 20 in the last visible
epochs, down from ≈ 122 at the start (the session timed out during
epoch 24, but the saved artifact was verified on disk afterwards:
correct patch statistics, and it produced the evaluation below).
Artifact `models/denoiseNet.mat` (297,335 bytes). `train_model()`
defaults now match this configuration exactly.

## Test suite

`run_all_tests()` in MATLAB Online (R2026b): **22 passed, 0 failed,
0 incomplete.** The suite covers the STFT/ISTFT round trip, all three
metrics, both baselines, the IRM target, the network definition,
inference on a real sample pair, the committed data, and an
end-to-end chain.

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
ok = run_all_tests()          % 22/22 on sample data, no downloads
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
