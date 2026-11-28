# Objective evaluation

Data: VoiceBank-DEMAND standard test set (data/voicebank). Files evaluated: 824 (combined from 4 chunked runs). Metrics are means over files (segmental SNR and STOI: higher is better; LSD: lower is better).

| Method | Segmental SNR (dB) | LSD (dB) | STOI | SI-SDR (dB) |
|---|---|---|---|---|
| Noisy | 1.52 | 14.50 | 0.921 | 8.45 |
| SpectralSubtraction | 2.08 | 13.60 | 0.922 | 8.87 |
| Wiener | 2.99 | 12.71 | 0.919 | 9.62 |
| LogMMSE | 2.82 | 12.72 | 0.920 | 9.52 |
| CNN | 5.63 | 11.90 | 0.920 | 14.20 |

Per-SNR breakdown: `metrics_by_snr.csv`. Per-noise-type breakdown is not reported: the distributed corpus files carry no per-file noise-type label, and none is inferred.

Segmental-SNR improvement of the CNN over the noisy input: 4.12 dB.

