#!/usr/bin/env python3
"""Synthesize FLEURS eval texts with Kinya-Flex-TTS (DeepKIN).

Requires:
  bash scripts/setup_deepkin.sh
  baselines at $BASELINES/kinya_flex_tts/kinya_flex_tts_base_trained.pt

  python scripts/eval_flex_fleurs.py --split test --device cuda --speaker 0
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
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
        "--checkpoint",
        default=os.environ.get("BASELINES", "/project/community/rmwisene/tts_baselines")
        + "/kinya_flex_tts/kinya_flex_tts_base_trained.pt",
    )
    p.add_argument(
        "--out_dir",
        default=os.environ.get("BASELINES", "/project/community/rmwisene/tts_baselines")
        + "/eval_fleurs/kinya_flex_tts",
    )
    p.add_argument("--speaker", type=int, default=0, help="0=F1, 1=F2, 2=Male")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--deepkin_root",
        default=os.environ.get(
            "DEEPKIN_ROOT",
            "/project/community/rmwisene/code/ac-ai-models/DeepKIN-AgAI",
        ),
        help="Path to DeepKIN-AgAI (added to PYTHONPATH if needed)",
    )
    args = p.parse_args()

    deepkin = Path(args.deepkin_root)
    if deepkin.is_dir() and str(deepkin) not in sys.path:
        sys.path.insert(0, str(deepkin))

    import torch
    import torchaudio
    from deepkin.data.kinya_norm import text_to_sequence
    from deepkin.models.flex_tts import FlexKinyaTTS
    from deepkin.modules.tts_commons import intersperse

    manifest = Path(args.manifest_dir) / f"{args.split}.tsv"
    if not manifest.is_file():
        raise SystemExit(
            f"Missing {manifest}. Run: python scripts/clean_fleurs_eval.py"
        )
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(
            f"Missing {ckpt}. Run: bash scripts/download_baselines.sh"
        )

    rows = read_manifest(manifest)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    device_s = args.device
    if device_s == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — using CPU")
        device_s = "cpu"
    device = torch.device(device_s if device_s != "cuda" else "cuda:0")

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Kinya-Flex from {ckpt} on {device} (speaker={args.speaker}) …")
    model = FlexKinyaTTS.from_pretrained(device, str(ckpt))
    model.eval()
    sample_rate = 24000

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
                "speaker",
            ],
            delimiter="\t",
        )
        w.writeheader()

        for i, row in enumerate(rows, 1):
            utt_id = row["id"]
            text = row["text"]
            hyp = out_dir / f"{utt_id}.wav"

            text_ids = intersperse(text_to_sequence(text, norm=True), 0)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                audio = model(text_ids, args.speaker, speed=args.speed)
            if device.type == "cuda":
                torch.cuda.synchronize()
            synth_time = time.perf_counter() - t0

            if not isinstance(audio, torch.Tensor):
                audio = torch.as_tensor(audio)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            audio = audio.detach().cpu().float()
            dur = float(audio.shape[-1] / sample_rate)
            rtf = synth_time / dur if dur > 0 else float("nan")

            torchaudio.save(str(hyp), audio, sample_rate)
            w.writerow(
                {
                    "id": utt_id,
                    "text": text,
                    "ref_audio": row.get("audio_path", ""),
                    "hyp_audio": str(hyp.resolve()),
                    "duration_s": f"{dur:.3f}",
                    "synth_time_s": f"{synth_time:.4f}",
                    "rtf": f"{rtf:.4f}",
                    "speaker": str(args.speaker),
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
