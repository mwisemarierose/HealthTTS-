#!/usr/bin/env python3
"""Convert ASR-expanded Kidaw’ida processed/ TSVs → TTS CSV manifests.

Input (Drive tar layout):
  …/tts_data_cleaning/processed/kidawida/manifests/{train,dev,test}_processed.tsv
  …/tts_data_cleaning/processed/kidawida/audio/{train,dev,test}/*.wav

Output (same schema as Kin pool / prepare_kidawida_tts_corpus):
  …/tts_data_cleaning/kidawida_tts_prepared/manifests/{train,dev,test}.csv

  python scripts/convert_kidawida_asr_processed_to_tts_manifests.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SCHEMA = ("utt_id", "audio_path", "text", "speaker_id", "source", "duration", "split")


def convert_split(
    tsv_path: Path,
    root: Path,
    split: str,
    source: str,
) -> list[dict]:
    rows_out: list[dict] = []
    missing = 0
    with tsv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, r in enumerate(reader, 1):
            rel = (r.get("audio_path") or r.get("path") or "").strip()
            if not rel:
                continue
            # paths in TSV are relative to kidawida root
            audio = root / rel
            if not audio.is_file():
                # sometimes path is mp3 name only under audio/split/
                stem = Path(rel).name
                if not stem.endswith(".wav"):
                    stem = Path(stem).with_suffix(".wav").name
                audio = root / "audio" / split / stem
            if not audio.is_file():
                missing += 1
                continue
            text = (r.get("transcript") or r.get("sentence") or "").strip()
            if not text:
                continue
            dur = float(r.get("duration_sec") or r.get("duration") or 0.0)
            spk = (r.get("client_id") or "dav_spk_unknown").strip()
            utt = Path(audio).stem
            rows_out.append(
                {
                    "utt_id": utt,
                    "audio_path": str(audio.resolve()),
                    "text": text,
                    "speaker_id": spk,
                    "source": source,
                    "duration": f"{dur:.4f}",
                    "split": split,
                }
            )
    print(f"  {split}: kept={len(rows_out)} missing_audio={missing} from {tsv_path.name}")
    return rows_out


def hours(rows: list[dict]) -> float:
    return sum(float(r["duration"]) for r in rows) / 3600.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_root",
        type=Path,
        default=Path(
            "/project/community/rmwisene/tts_data_cleaning/processed/kidawida"
        ),
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path(
            "/project/community/rmwisene/tts_data_cleaning/kidawida_tts_prepared"
        ),
    )
    ap.add_argument("--source", default="kidawida_asr_expanded")
    args = ap.parse_args()

    man_in = args.in_root / "manifests"
    man_out = args.out_dir / "manifests"
    man_out.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for split in ("train", "dev", "test"):
        tsv = man_in / f"{split}_processed.tsv"
        if not tsv.is_file():
            raise SystemExit(f"Missing {tsv}")
        rows = convert_split(tsv, args.in_root, split, args.source)
        out_csv = man_out / f"{split}.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SCHEMA)
            w.writeheader()
            w.writerows(rows)
        all_stats[split] = {
            "n": len(rows),
            "hours": round(hours(rows), 4),
            "path": str(out_csv),
        }
        print(f"  wrote {out_csv}  ({all_stats[split]['hours']:.2f} h)")

    meta = {
        "in_root": str(args.in_root),
        "out_dir": str(args.out_dir),
        "source": args.source,
        "splits": all_stats,
    }
    (args.out_dir / "convert_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
