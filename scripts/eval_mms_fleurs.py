#!/usr/bin/env python3
"""Run MMS-TTS Kin synthesis on eval manifests (FLEURS TSV or Kin-pool CSV).

FLEURS / Kidaw’ida TSV columns: id, text, audio_path
Kin pool CSV columns: utt_id, text, audio_path, speaker_id (optional)

  # Kin training-pool dev (E1 checkpoint = output_dir after save_pretrained)
  python scripts/eval_mms_fleurs.py \\
    --manifest_dir /project/community/rmwisene/tts_data_cleaning/kinyarwanda_tts_mbaza_afrispeak/manifests \\
    --split dev \\
    --model_dir /project/community/rmwisene/tts_baselines/finetune_e1_mms_kin \\
    --out_dir /project/community/rmwisene/tts_baselines/eval_kin_dev/mms_e1

  # FLEURS test (zero-shot baseline)
  python scripts/eval_mms_fleurs.py --split test --limit 20
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path


SOURCE_TO_SPK = {
    "afrispeak_female": 0,
    "mbaza_tts": 1,
    "afrispeak_male": 2,
}


def resolve_speaker_id(row: dict) -> int:
    """Kin pool CSV may store speaker_id as 'mbaza_spk0' — map via source or int."""
    raw = row.get("speaker_id", "")
    if raw is not None and str(raw).strip() != "":
        s = str(raw).strip()
        try:
            return int(s)
        except ValueError:
            pass
        # e.g. mbaza_spk0 / afrispeak_female_spk1
        low = s.lower()
        if "afrispeak_female" in low or low.startswith("af_f"):
            return 0
        if "afrispeak_male" in low or low.startswith("af_m"):
            return 2
        if "mbaza" in low:
            return 1
    source = (row.get("source") or "").strip()
    return SOURCE_TO_SPK.get(source, 0)


def read_manifest(path: Path) -> list[dict]:
    """Normalize rows to id, text, audio_path, speaker_id (int)."""
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
        "Kin pool: manifests under tts_data_cleaning/kinyarwanda_tts_mbaza_afrispeak/manifests"
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
    p.add_argument(
        "--speaker_id",
        type=int,
        default=-1,
        help="Force speaker id for all utts (-1 = use row / default 0)",
    )
    args = p.parse_args()

    import torch
    import scipy.io.wavfile
    from transformers import VitsModel, AutoTokenizer

    manifest = resolve_manifest(Path(args.manifest_dir), args.split)
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

    print(f"Manifest: {manifest}  (n={len(rows)})")
    print(f"Loading {model_dir} on {device} …")
    model = VitsModel.from_pretrained(str(model_dir)).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model.eval()
    rate = int(model.config.sampling_rate)
    multi_spk = int(getattr(model.config, "num_speakers", 1) or 1) > 1

    meta_path = out_dir / "synthesis.tsv"
    with meta_path.open("w", encoding="utf-8", newline="") as mf:
        w = csv.DictWriter(
            mf,
            fieldnames=[
                "id",
                "text",
                "ref_audio",
                "hyp_audio",
                "speaker_id",
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
            spk = args.speaker_id if args.speaker_id >= 0 else int(row["speaker_id"])
            hyp = out_dir / f"{utt_id}.wav"
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if multi_spk:
                inputs["speaker_id"] = torch.tensor([spk], device=device)
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
                    "speaker_id": spk,
                    "duration_s": f"{dur:.3f}",
                    "synth_time_s": f"{synth_time:.4f}",
                    "rtf": f"{rtf:.4f}",
                }
            )
            if i % 10 == 0 or i == len(rows):
                print(
                    f"  [{i}/{len(rows)}] {utt_id} spk={spk} "
                    f"({dur:.2f}s audio, {synth_time:.3f}s synth, RTF={rtf:.3f})"
                )

    print(f"Wrote synth → {out_dir}")
    print(f"Wrote meta  → {meta_path}")


if __name__ == "__main__":
    main()
