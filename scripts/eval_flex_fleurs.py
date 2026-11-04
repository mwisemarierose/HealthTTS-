#!/usr/bin/env python3
"""Synthesize eval texts with Kinya-Flex-TTS (DeepKIN).

Supports FLEURS/Kidaw’ida TSV (id, text, audio_path) and Kin-pool CSV
(utt_id, text, audio_path, source / speaker_id).

Requires:
  bash scripts/setup_deepkin.sh
  checkpoint .pt under $BASELINES

  # Kin pool DEV (E2)
  python scripts/eval_flex_fleurs.py \\
    --manifest_dir .../kinyarwanda_tts_mbaza_afrispeak/manifests \\
    --split dev \\
    --checkpoint .../finetune_e2_flex_kin/kinya_flex_tts_e2.pt \\
    --out_dir .../eval_kin_dev/flex_e2 \\
    --speaker -1
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Flex: 0=Female1, 1=Female2, 2=Male — align Kin sources similarly to MMS eval
SOURCE_TO_SPK = {
    "afrispeak_female": 0,
    "mbaza_tts": 1,
    "afrispeak_male": 2,
}


def resolve_speaker_id(row: dict, default: int = 0) -> int:
    raw = row.get("speaker_id", "")
    if raw is not None and str(raw).strip() != "":
        s = str(raw).strip()
        try:
            return int(s)
        except ValueError:
            pass
        low = s.lower()
        if "afrispeak_female" in low or low.startswith("af_f"):
            return 0
        if "afrispeak_male" in low or low.startswith("af_m"):
            return 2
        if "mbaza" in low:
            return 1
    source = (row.get("source") or "").strip()
    return SOURCE_TO_SPK.get(source, default)


def read_manifest(path: Path) -> list[dict]:
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(encoding="utf-8", newline="") as f:
        raw = list(csv.DictReader(f, delimiter=delim))
    rows = []
    for r in raw:
        utt_id = (r.get("id") or r.get("utt_id") or "").strip()
        text = (r.get("text") or "").strip()
        audio = (r.get("audio_path") or r.get("path") or "").strip()
        if not utt_id or not text:
            continue
        rows.append(
            {
                "id": utt_id,
                "text": text,
                "audio_path": audio,
                "speaker_id": resolve_speaker_id(r),
            }
        )
    return rows


def resolve_manifest(manifest_dir: Path, split: str) -> Path:
    for name in (f"{split}.tsv", f"{split}.csv"):
        p = manifest_dir / name
        if p.is_file():
            return p
    raise SystemExit(
        f"Missing {manifest_dir}/{split}.tsv|csv\n"
        "FLEURS: python scripts/clean_fleurs_eval.py | "
        "Kidaw’ida: python scripts/prepare_kidawida_eval_manifest.py | "
        "Kin pool: tts_data_cleaning/kinyarwanda_tts_mbaza_afrispeak/manifests"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest_dir",
        default="/project/community/rmwisene/datasets/fleurs_kinyarwanda/processed/eval",
    )
    p.add_argument(
        "--split",
        default="test",
        help="Manifest basename without extension (test/dev)",
    )
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
    p.add_argument(
        "--speaker",
        type=int,
        default=0,
        help="0=F1, 1=F2, 2=Male; -1 = use row speaker_id/source (Kin pool)",
    )
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

    import numpy as np
    import soundfile as sf
    import torch
    from deepkin.data.kinya_norm import text_to_sequence
    from deepkin.models.flex_tts import FlexKinyaTTS
    from deepkin.modules.tts_commons import intersperse

    manifest = resolve_manifest(Path(args.manifest_dir), args.split)
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

    spk_msg = "per-utt" if args.speaker < 0 else str(args.speaker)
    print(f"Manifest: {manifest}  (n={len(rows)})")
    print(f"Loading Kinya-Flex from {ckpt} on {device} (speaker={spk_msg}) …")
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
            speaker = int(row["speaker_id"]) if args.speaker < 0 else args.speaker

            text_ids = intersperse(text_to_sequence(text, norm=True), 0)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                audio = model(text_ids, speaker, speed=args.speed)
            if device.type == "cuda":
                torch.cuda.synchronize()
            synth_time = time.perf_counter() - t0

            if not isinstance(audio, torch.Tensor):
                audio = torch.as_tensor(audio)
            audio = audio.detach().cpu().float().numpy()
            if audio.ndim > 1:
                audio = audio.squeeze()
            audio = np.asarray(audio, dtype=np.float32)
            dur = float(audio.shape[-1] / sample_rate)
            rtf = synth_time / dur if dur > 0 else float("nan")

            sf.write(str(hyp), audio, sample_rate)
            w.writerow(
                {
                    "id": utt_id,
                    "text": text,
                    "ref_audio": row.get("audio_path", ""),
                    "hyp_audio": str(hyp.resolve()),
                    "duration_s": f"{dur:.3f}",
                    "synth_time_s": f"{synth_time:.4f}",
                    "rtf": f"{rtf:.4f}",
                    "speaker": str(speaker),
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
