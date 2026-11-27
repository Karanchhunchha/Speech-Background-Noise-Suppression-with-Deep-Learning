# Sample data provenance

The 8 clean/noisy pairs in this folder are a small excerpt of the
**VoiceBank-DEMAND** corpus, included so the demo and the unit tests run
on a fresh clone with no download:

- C. Valentini-Botinhao, "Noisy speech database for training speech
  enhancement algorithms and TTS models", Edinburgh DataShare, 2017.
  DOI: [10.7488/ds/2117](https://doi.org/10.7488/ds/2117)
- Licence: **CC BY 4.0** (attribution required — this file is that
  attribution). The DEMAND noise recordings are CC BY-SA 3.0.
- The files are the corpus's standard test-set items (speakers p232 and
  p257, unseen in training), 16 kHz versions identical to the
  JacobLinCool/VoiceBank-DEMAND-16k mirror of the DataShare release.

File names are the corpus's own (`p232_001.wav`, etc.). The full test
set (824 pairs) and the training pairs are fetched by `download_data()`
and are not committed (size).
