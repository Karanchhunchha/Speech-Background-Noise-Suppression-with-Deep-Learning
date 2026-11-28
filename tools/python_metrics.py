#!/usr/bin/env python3
"""Offline PESQ / DNSMOS scoring for enhanced test clips.

WHY OFFLINE: PESQ (ITU-T P.862) needs a licensed reference
implementation and is not available inside this project's MATLAB
environment, so it is computed here, outside the toolbox-free MATLAB
pipeline, on enhanced .wav files exported by
    evaluate_models('SaveEnhancedDir', 'reports/enhanced_subset')
Run that first (any subset size; the report states N honestly).

Usage:
    pip install pesq soundfile numpy        # PESQ (wideband, 16 kHz)
    python tools/python_metrics.py reports/enhanced_subset

Input folder layout (as written by evaluate_models):
    <id>_clean.wav, <id>_noisy.wav, <id>_spectralsubtraction.wav,
    <id>_wiener.wav, <id>_logmmse.wav, <id>_cnn.wav
Output: reports/pesq_results.csv with per-file and mean PESQ-WB.
If the `pesq` package is unavailable the script says so and exits
without writing any numbers — scores are never approximated silently.
DNSMOS is intentionally not bundled: it requires downloading a
third-party ONNX model; see reports/README note in EVALUATION docs.
"""
import csv
import sys
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
    from pesq import pesq
except ImportError as exc:
    sys.exit(f"Missing dependency ({exc.name}). Install: pip install pesq soundfile numpy")


def main(folder: str) -> None:
    folder = Path(folder)
    methods = ["noisy", "spectralsubtraction", "wiener", "logmmse", "cnn"]
    rows = []
    for clean_path in sorted(folder.glob("*_clean.wav")):
        clip_id = clean_path.name[: -len("_clean.wav")]
        ref, fs = sf.read(clean_path)
        if fs != 16000:
            print(f"skip {clip_id}: sample rate {fs} != 16000 (PESQ-WB needs 16 kHz)")
            continue
        for method in methods:
            deg_path = folder / f"{clip_id}_{method}.wav"
            if not deg_path.exists():
                continue
            deg, _ = sf.read(deg_path)
            n = min(len(ref), len(deg))
            score = pesq(16000, ref[:n], deg[:n], "wb")
            rows.append({"file": clip_id, "method": method, "pesq_wb": round(float(score), 3)})
    if not rows:
        sys.exit("No scorable pairs found — run evaluate_models with 'SaveEnhancedDir' first.")
    out = Path("reports/pesq_results.csv")
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "method", "pesq_wb"])
        writer.writeheader()
        writer.writerows(rows)
    for method in methods:
        vals = [r["pesq_wb"] for r in rows if r["method"] == method]
        if vals:
            print(f"{method:22s} PESQ-WB mean {np.mean(vals):.3f}  (N={len(vals)})")
    print(f"Wrote {out} ({len(rows)} scores)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reports/enhanced_subset")
