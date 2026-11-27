# Objective evaluation

Data: VoiceBank-DEMAND standard test set (data/voicebank). Files evaluated: 824 (combined from 2 chunked runs). Metrics are means over files (segmental SNR and STOI: higher is better; LSD: lower is better).

| Method | Segmental SNR (dB) | LSD (dB) | STOI |
|---|---|---|---|
| Noisy | 1.52 | 14.50 | 0.921 |
| SpectralSubtraction | 2.08 | 13.60 | 0.922 |
| Wiener | 2.99 | 12.71 | 0.919 |
| CNN | 5.88 | 11.37 | 0.912 |

Segmental-SNR improvement of the CNN over the noisy input: 4.36 dB.
