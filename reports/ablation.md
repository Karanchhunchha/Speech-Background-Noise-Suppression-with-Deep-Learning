# Phase-1 ablation — closing the STOI gap

Baseline (frozen, `reports/baseline_v1.csv`): on the full 824-file
test set the shipped CNN scores SegSNR 5.88 dB / LSD 11.37 dB /
STOI 0.912 against the noisy input's 0.921 — a STOI gap of −0.009.
Phase 1 tries to close that gap without giving up the SNR gain.
Nothing here is tuned on the test set: configurations are ranked on
training-corpus validation data and the winner is evaluated exactly
once on the full test set.

## Step 1a — inference-time mask options (no retraining)

Validation: 58 files of training speaker **p228** (all that exist for
that speaker in the extracted shard). Caveat, stated plainly: the
shipped model was trained on data that includes p228, so absolute
validation values are optimistic; the table is used only to *rank*
configurations. Shipped model, `denoiseWithNet` options
`MaskFloor` / `MaskExponent`, run 27 Nov 2026 in MATLAB Online:

| Config | SegSNR (dB) | LSD (dB) | STOI |
|---|---|---|---|
| noisy (reference) | 0.76 | 17.30 | 0.8348 |
| floor 0.00, exp 1.0 (shipped) | 4.05 | 13.12 | 0.8194 |
| floor 0.00, exp 0.7 | 3.83 | 13.88 | 0.8282 |
| floor 0.05, exp 1.0 | 4.02 | 13.11 | 0.8196 |
| floor 0.05, exp 0.7 | 3.81 | 13.88 | 0.8283 |
| floor 0.10, exp 1.0 | 3.96 | 13.13 | 0.8201 |
| floor 0.10, exp 0.7 | 3.77 | 13.88 | 0.8284 |
| floor 0.20, exp 1.0 | 3.72 | 13.30 | 0.8225 |
| floor 0.20, exp 0.7 | 3.66 | 13.93 | 0.8291 |

**Reading:** softer suppression (exponent 0.7, higher floor)
monotonically improves validation STOI, but the best inference-only
configuration (floor 0.20, exp 0.7, STOI 0.8291) is still 0.0057
*below* the noisy reference and gives up 0.39 dB of segmental SNR
against the shipped configuration. Inference-time tuning alone does
not close the gap → Phase 1b (retraining: more speakers/data and a
perceptual/spectral loss) is the real lever, ranked on a genuinely
speaker-disjoint validation set built from an extra training shard.

## Step 1b — retraining (in progress)

*(Results appended as runs complete; failed runs are recorded too.)*
