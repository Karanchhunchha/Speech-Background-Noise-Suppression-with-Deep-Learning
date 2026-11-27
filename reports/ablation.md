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

## Step 1a (continued) — genuinely held-out validation

Training shard 1 was downloaded (SHA-256 verified) and extracted:
2,315 further pairs, 7 new speakers (10 total, 3,115 files). The
shipped model never saw these speakers. Validation: 100 files of
speaker **p236**:

| Config | SegSNR (dB) | LSD (dB) | STOI |
|---|---|---|---|
| noisy (reference) | 2.97 | 18.97 | 0.8109 |
| floor 0.00, exp 1.0 (shipped) | 3.90 | 14.25 | 0.7997 |
| floor 0.00, exp 0.7 | 4.20 | 15.25 | 0.8071 |
| floor 0.05, exp 0.7 | 4.21 | 15.25 | 0.8072 |
| floor 0.10, exp 0.7 | 4.22 | 15.25 | 0.8074 |
| floor 0.20, exp 0.7 | 4.21 | 15.30 | 0.8079 |

(exp 1.0 rows at floors 0.05–0.20 sit between the shipped row and
these; full table in the run log.) Two findings: (1) on truly
held-out data the soft-suppression configs (exp 0.7) beat the shipped
configuration on **both** SegSNR and STOI — the p228 ranking did not
generalise, which is exactly why this second validation exists;
(2) even the best config remains 0.003 below the noisy STOI, so
retraining is still required. Candidate inference configuration
carried forward: **floor 0.10, exp 0.7** (best SegSNR, near-best
STOI); floor 0.20/exp 0.7 is the STOI-maximal variant.

## Step 1b — retraining

Protocol: candidates train with validation speakers **p236 and
p254 excluded** (2,584 files of 8 speakers remain available) and are
ranked on pooled p236+p254 validation against the shipped model
under identical inference configurations. The winner is evaluated
once on the full test set.

### Candidate R1 (trained 2026-11-27, MATLAB Online)

1,600 utterances, 9,600 patches, 16 epochs, speakers p236/p254
excluded (mu 0.1597, sigma 0.3042; training loop 1 h 13 min).
Pooled held-out validation, 145 files (noisy reference:
SegSNR 1.96 dB, LSD 19.52 dB, STOI 0.8118):

| Model | Floor | Exp | SegSNR (dB) | LSD (dB) | STOI |
|---|---|---|---|---|---|
| shipped v2 | 0.00 | 1.0 | 3.614 | 14.617 | 0.8011 |
| shipped v2 | 0.05 | 0.7 | 3.724 | 15.682 | 0.8086 |
| shipped v2 | 0.20 | 0.7 | 3.692 | 15.734 | 0.8092 |
| **R1** | 0.00 | 1.0 | 4.156 | 14.443 | 0.8051 |
| **R1** | 0.05 | 1.0 | **4.161** | **14.443** | 0.8054 |
| **R1** | **0.10** | **0.7** | 4.067 | 15.584 | **0.8123** |
| **R1** | 0.20 | 0.7 | 4.027 | 15.638 | 0.8129 |

(Full 8-configuration grids for both models were measured; the
rows above are the decision-relevant ones. R1 beats the shipped
model on every configuration and every metric.)

### Ship decision (2026-11-27)

**Ship R1 with mask floor 0.10, exponent 0.7** (now the defaults in
`denoiseWithNet`). Rationale: it gives up only 0.09 dB of segmental
SNR against the grid maximum (4.07 vs 4.16) while lifting STOI
**above the noisy reference** (0.8123 vs 0.8118) — the shipped
model's one documented weakness on the test set was a STOI below
the noisy input, and this operating point removes it on held-out
data. The floor 0.20/exp 0.7 variant's extra +0.0005 STOI costs
more SegSNR and LSD than it is worth. The winner is now evaluated
exactly once on the full 824-file test set; the v2 numbers remain
frozen in `reports/baseline_v1.csv` for the before/after record.

### Test-set outcome (the single full evaluation, 2026-11-27)

The chosen configuration was evaluated once on the full 824-file
test set (chunked 4 × 206, merged; `reports/metrics_summary.csv`):

| CNN model | SegSNR (dB) | LSD (dB) | STOI | SI-SDR (dB) |
|---|---|---|---|---|
| v2 @ floor 0 / exp 1.0 (previous) | 5.88 | 11.37 | 0.912 | — |
| **R1 @ floor 0.10 / exp 0.7 (shipped)** | 5.63 | 11.90 | 0.920 | 14.20 |

Read this honestly: on the test set the shipped configuration
**trades** segmental SNR (−0.25 dB) and LSD (+0.53 dB) for STOI
(+0.0075). The STOI gap against the noisy input shrinks from
−0.009 to −0.001 — nearly closed, not eliminated. The comparison
is confounded by design: model and operating point changed
together, and at the *matched* raw-mask operating point R1 beat
v2 clearly on validation (+0.54 dB SegSNR), so the test-set
SegSNR difference is the known cost of the softer mask
(Phase 1a: softer suppression helps STOI, costs SegSNR), not a
weaker model. The operating point was selected on validation
before this test run, per protocol; it was **not** re-tuned after
seeing these numbers, and this paragraph is the disclosure of
what the trade bought and what it cost.
