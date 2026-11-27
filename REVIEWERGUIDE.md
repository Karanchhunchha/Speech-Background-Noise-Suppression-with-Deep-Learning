# Reviewer guide

Thank you for reviewing this submission for **Project #193 (Speech
Background Noise Suppression with Deep Learning)**. This page is the
fastest path to verifying the work.

## The 5-minute path

```matlab
main()                   % single entry point: tests → demo → evaluation
```

or stage by stage:

```matlab
ok = run_all_tests()     % 25 unit tests, sample data only, no downloads
demo_denoise()           % before/after .wav files + spectrograms in results/demo/
```

Then read `reports/EVALUATION.md` for the verified objective results
(full 824-file test set) and the README's results table.

## The full path (reproduce everything)

```matlab
download_data()          % VoiceBank-DEMAND: 824 test pairs + 800 training pairs
train_model()            % builds patches, trains the CNN, saves models/denoiseNet.mat
summary = evaluate_models()   % rewrites reports/metrics_*.csv and reports/EVALUATION.md
% (For very long evaluations, EVALUATE_MODELS accepts 'StartIndex'/'MaxFiles'
%  chunks; MERGE_EVALUATION combines the chunk outputs into the final reports.)
```

## Where the evidence lives

| Claim | Evidence |
|---|---|
| The pipeline runs end to end | `run_all_tests()` — 25 tests incl. an end-to-end chain on a real sample pair |
| The CNN beats the classical baselines on segmental SNR and LSD (STOI is 0.009 below the noisy input — see the README) | `reports/EVALUATION.md`, `reports/metrics_summary.csv` (means over the full test set), `reports/metrics_per_file.csv` (every file, auditable) |
| The metrics are trustworthy | `tests/test_Metrics.m`: STOI(x, x) = 1, STOI falls as noise rises, LSD(x, x) = 0, segmental SNR of a perfect estimate = clip maximum. The STOI implementation was additionally cross-validated against the reference algorithm (pystoi): agreement within 0.0005 on all 8 sample pairs — see the `stoiScore` header for the window-length bug this caught |
| No test-set leakage | Training pairs come only from the corpus's 28-speaker training set; the test set's two speakers (p232, p257) never appear in training. `download_data.m` keeps the splits in separate folders and `prepare_training_data.m` reads only the training folders |
| The demo is real | `demo_denoise()` regenerates every demo .wav from the committed sample inputs with the committed model |

## Design decisions worth knowing

- **Toolbox-free DSP and metrics.** STFT/ISTFT, the FFT resampler and
  STOI are implemented from their definitions in `src/`. Only the
  network itself needs the Deep Learning Toolbox. This was deliberate:
  the evaluation must be reproducible anywhere MATLAB runs.
- **Percentile noise floor.** Both classical baselines estimate noise
  from the 10th percentile of each frequency bin over time, because
  VoiceBank-DEMAND mixtures do not guarantee a noise-only head segment.
- **Small network on purpose.** ~24k parameters, trained on 800
  utterances (4,800 patches, 24 epochs, ~55 minutes on CPU) — the
  whole pipeline is reproducible on a student laptop without a GPU.
  Options to scale up are exposed.
- **Honest reporting.** The evaluation script writes per-file metrics,
  not just means, and the README states the limitations (noisy phase,
  compact model, clean-room STOI) next to the results.
