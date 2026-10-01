#!/usr/bin/env python3
"""Smoke-load facebook/mms-tts-kin and synthesize one Kinyarwanda sentence.

Usage on Orchard (GPU node preferred):
  conda activate healthtts
  export BASELINES=/project/community/rmwisene/tts_baselines
  python scripts/smoke_mms_tts_kin.py
  python scripts/smoke_mms_tts_kin.py --text "Muraho, amakuru?" --out /tmp/mms_kin_smoke.wav
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default=os.environ.get(
            "BASELINES", "/project/community/rmwisene/tts_baselines"
        )
        + "/mms_tts_kin",
    )
    parser.add_argument(
        "--text",
        default="Muraho. Amakuru yawe?",
        help="Kinyarwanda text to synthesize",
    )
    parser.add_argument(
        "--out",
        default="/tmp/mms_tts_kin_smoke.wav",
        help="Output wav path",
    )
    args = parser.parse_args()

    import torch
    from transformers import VitsModel, AutoTokenizer
    import scipy.io.wavfile

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {model_dir}. Run: bash scripts/download_baselines.sh"
        )

    print(f"Loading MMS-TTS Kin from {model_dir} …")
    model = VitsModel.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model.eval()

    inputs = tokenizer(args.text, return_tensors="pt")
    with torch.no_grad():
        waveform = model(**inputs).waveform

    audio = waveform.squeeze().cpu().numpy()
    rate = int(model.config.sampling_rate)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(str(out), rate=rate, data=audio)
    print(f"Wrote {out} ({audio.shape[-1] / rate:.2f}s @ {rate} Hz)")


if __name__ == "__main__":
    main()
