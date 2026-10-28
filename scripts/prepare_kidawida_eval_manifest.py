#!/usr/bin/env python3
"""Build FLEURS-style eval TSVs from ASR Kidaw’ida processed manifests.

Zero-shot baseline eval (B0/B1) uses test (primary) and optionally dev.
Does NOT modify the ASR corpus.

  python scripts/prepare_kidawida_eval_manifest.py

Writes:
  /project/community/rmwisene/datasets/kidawida_asr_drive/processed/kidawida/eval/
    test.tsv  dev.tsv  (and train.tsv if present)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

OUT_FIELDS = ["id", "split", "audio_path", "text", "duration_s"]


def resolve_audio(raw: str, audio_root: Path, data_root: Path) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    if p.is_file():
        return p.resolve()
    for base in (audio_root, data_root, data_root / "processed" / "kidawida"):
        cand = base / raw
        if cand.is_file():
            return cand.resolve()
        cand = base / Path(raw).name
        if cand.is_file():
            return cand.resolve()
    # relative like audio/test/foo.wav under processed/kidawida
    for base in (data_root / "processed" / "kidawida", data_root):
        cand = base / raw
        if cand.is_file():
            return cand.resolve()
    return None


def load_asr_tsv(path: Path, split: str, audio_root: Path, data_root: Path) -> list[dict]:
    rows_out: list[dict] = []
    with path.open(encoding="utf-8-sig", newline="", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, r in enumerate(reader):
            text = (r.get("transcript") or r.get("sentence") or r.get("text") or "").strip()
            audio_raw = (r.get("audio_path") or r.get("path") or "").strip()
            if not text or not audio_raw:
                continue
            audio = resolve_audio(audio_raw, audio_root, data_root)
            if audio is None:
                continue
            stem = Path(audio_raw).stem or Path(audio).stem
            utt_id = f"dav_{split}_{stem}"
            dur = (r.get("duration_sec") or r.get("duration_s") or r.get("duration") or "").strip()
            rows_out.append(
                {
                    "id": utt_id,
                    "split": split,
                    "audio_path": str(audio),
                    "text": text,
                    "duration_s": dur,
                }
            )
    return rows_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root",
        type=Path,
        default=Path("/project/community/rmwisene/datasets/kidawida_asr_drive"),
    )
    ap.add_argument(
        "--manifest_dir",
        type=Path,
        default=None,
        help="Default: <data_root>/processed/kidawida/manifests",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Default: <data_root>/processed/kidawida/eval",
    )
    ap.add_argument(
        "--splits",
        default="test,dev",
        help="Comma-separated: test,dev[,train]",
    )
    args = ap.parse_args()

    data_root = args.data_root
    man_dir = args.manifest_dir or (data_root / "processed" / "kidawida" / "manifests")
    out_dir = args.out_dir or (data_root / "processed" / "kidawida" / "eval")
    audio_root = data_root / "processed" / "kidawida" / "audio"

    if not man_dir.is_dir():
        raise SystemExit(f"Missing manifests: {man_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    name_map = {
        "train": "train_processed.tsv",
        "dev": "dev_processed.tsv",
        "test": "test_processed.tsv",
    }

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        src = man_dir / name_map.get(split, f"{split}_processed.tsv")
        if not src.is_file():
            print(f"SKIP {split}: missing {src}")
            continue
        rows = load_asr_tsv(src, split, audio_root, data_root)
        out = out_dir / f"{split}.tsv"
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=OUT_FIELDS, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"{split}: {len(rows)} → {out}")

    print(f"Done. Use --manifest_dir {out_dir} with eval_mms / eval_flex.")


if __name__ == "__main__":
    main()
