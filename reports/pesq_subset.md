# PESQ (offline, subset)

PESQ (ITU-T P.862, wideband) cannot run inside this project's
MATLAB environment, so it was computed offline with
`tools/python_metrics.py` (the `pesq` Python package) on enhanced
clips exported by
`evaluate_models('MaxFiles', 8, 'SaveEnhancedDir', ...)`.

**Sample: N = 8 files** — the first 8 files of the standard test
set (p232_001, 002, 003, 005, 006, 007, 009, 010), enhanced by the
shipped R1 model at its shipped defaults. This is a subset result;
the full-set metrics (SegSNR / LSD / STOI / SI-SDR, N = 824) are in
`metrics_summary.csv`.

| Method | PESQ-WB (mean, N = 8) |
|---|---|
| Noisy | 2.114 |
| SpectralSubtraction | 2.185 |
| Wiener | 2.247 |
| CNN (shipped) | **2.334** |

Per-file scores: `reports/pesq_results.csv` (32 scores).
The CNN ranks first on PESQ as it does on the full-set metrics.
