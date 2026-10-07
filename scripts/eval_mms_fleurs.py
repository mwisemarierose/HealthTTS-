#!/usr/bin/env python3
"""Run MMS-TTS Kin baseline synthesis on cleaned FLEURS eval texts.

FLEURS is evaluation-only — this does NOT train. It synthesizes from
cleaned manifests (test/dev) for listening / later ASR-based metrics.

  conda activate healthtts
  export BASELINES=/project/community/rmwisene/tts_baselines
  export HF_HOME=/project/community/rmwisene/.cache/huggingface

  # small smoke (20 utts)
  python scripts/eval_mms_fleurs.py --split test --limit 20

  # full test split
  python scripts/eval_mms_fleurs.py --split test
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest_dir",
        default="/project/community/rmwisene/datasets/fleurs_kinyarwanda/processed/eval",
    )
    p.add_argument("--split", default="test", choices=["test", "dev"])
    p.add_argument(
        "--model_dir",
        default=os.environ.get("BASELINES", "/project/community/rmwisene/tts_baselines")
        + "/mms_tts_kin",
    )
    p.add_argument(
        "--out_dir",
        default=os.environ.get("BASELINES", "/project/community/rmwisene/tts_baselines")
        + "/eval_fleurs/mms_tts_kin",
    )
    p.add_argument("--limit", type=int, default=0, help="0 = all rows")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import torch
    import scipy.io.wavfile
    from transformers import VitsModel, AutoTokenizer

    manifest = Path(args.manifest_dir) / f"{args.split}.tsv"
    if not manifest.is_file():
        raise SystemExit(
            f"Missing {manifest}. Run: python scripts/clean_fleurs_eval.py"
        )

    rows = read_manifest(manifest)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — using CPU")
        device = "cpu"

    print(f"Loading {model_dir} on {device} …")
    model = VitsModel.from_pretrained(str(model_dir)).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model.eval()
    rate = int(model.config.sampling_rate)

    meta_path = out_dir / "synthesis.tsv"
    with meta_path.open("w", encoding="utf-8", newline="") as mf:
        w = csv.DictWriter(
            mf,
            fieldnames=[
                "id",
                "text",
                "ref_audio",
                "hyp_audio",
                "duration_s",
                "synth_time_s",
                "rtf",
            ],
            delimiter="\t",
        )
        w.writeheader()

        for i, row in enumerate(rows, 1):
            utt_id = row["id"]
            text = row["text"]
            hyp = out_dir / f"{utt_id}.wav"
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                waveform = model(**inputs).waveform
            if device == "cuda":
                torch.cuda.synchronize()
            synth_time = time.perf_counter() - t0
            audio = waveform.squeeze().detach().cpu().numpy()
            scipy.io.wavfile.write(str(hyp), rate=rate, data=audio)
            dur = float(audio.shape[-1]) / rate
            rtf = synth_time / dur if dur > 0 else float("nan")
            w.writerow(
                {
                    "id": utt_id,
                    "text": text,
                    "ref_audio": row.get("audio_path", ""),
                    "hyp_audio": str(hyp.resolve()),
                    "duration_s": f"{dur:.3f}",
                    "synth_time_s": f"{synth_time:.4f}",
                    "rtf": f"{rtf:.4f}",
                }
            )
            if i % 10 == 0 or i == len(rows):
                print(
                    f"  [{i}/{len(rows)}] {utt_id} "
                    f"({dur:.2f}s audio, {synth_time:.3f}s synth, RTF={rtf:.3f})"
                )

    print(f"Wrote synth → {out_dir}")
    print(f"Wrote meta  → {meta_path}")


if __name__ == "__main__":
    main()
