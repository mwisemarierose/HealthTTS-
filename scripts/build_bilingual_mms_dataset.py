#!/usr/bin/env python3
"""Build MMS HF DatasetDict for bilingual Kin + Kidaw’ida fine-tunes (E3–E6).

E3/E4 balanced: all Kin train (~17.89 h) + Dav train subset ≈ same hours.
E5/E6 unbalanced: all Kin train + full Dav train.

Validation = Kin dev ∪ Dav dev (full).

  # E3 balanced
  python scripts/build_bilingual_mms_dataset.py --mode balanced --seed 42

  # E5 full Dav
  python scripts/build_bilingual_mms_dataset.py --mode full --out_dir .../bilingual_mms_hf_full
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

SPEAKER_MAP_KIN = {
    "afrispeak_female": 0,
    "mbaza_tts": 1,
    "afrispeak_male": 2,
}
# Kidaw’ida speakers start at 3 (override_speaker_embeddings grows the table)
DAV_SPEAKER_BASE = 3


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def hours(rows: list[dict]) -> float:
    return sum(float(r.get("duration") or 0.0) for r in rows) / 3600.0


def speaker_id_for(row: dict, dav_spk_map: dict[str, int]) -> int:
    src = (row.get("source") or "").strip()
    if src in SPEAKER_MAP_KIN:
        return SPEAKER_MAP_KIN[src]
    # Dav / other
    key = (row.get("speaker_id") or src or "dav_unknown").strip()
    if key not in dav_spk_map:
        dav_spk_map[key] = DAV_SPEAKER_BASE + len(dav_spk_map)
    return dav_spk_map[key]


def sample_by_hours(
    rows: list[dict], target_h: float, seed: int
) -> list[dict]:
    """Reproducible shuffle then take prefix until target hours (or all)."""
    rng = random.Random(seed)
    order = list(rows)
    rng.shuffle(order)
    out: list[dict] = []
    acc = 0.0
    for r in order:
        out.append(r)
        acc += float(r.get("duration") or 0.0) / 3600.0
        if acc >= target_h:
            break
    return out


def export_mms_hf(
    train_rows: list[dict],
    dev_rows: list[dict],
    out_dir: Path,
    dav_spk_map: dict[str, int],
    meta_extra: dict,
) -> None:
    from datasets import Audio, Dataset, DatasetDict, Features, Value

    def to_ds(rows: list[dict]) -> Dataset:
        return Dataset.from_dict(
            {
                "audio": [r["audio_path"] for r in rows],
                "text": [r["text"] for r in rows],
                "speaker_id": [speaker_id_for(r, dav_spk_map) for r in rows],
                "utt_id": [r.get("utt_id", "") for r in rows],
                "source": [r.get("source", "") for r in rows],
            },
            features=Features(
                {
                    "audio": Audio(sampling_rate=16000),
                    "text": Value("string"),
                    "speaker_id": Value("int64"),
                    "utt_id": Value("string"),
                    "source": Value("string"),
                }
            ),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = DatasetDict({"train": to_ds(train_rows), "validation": to_ds(dev_rows)})
    ds.save_to_disk(str(out_dir))
    meta = {
        "n_train": len(train_rows),
        "n_validation": len(dev_rows),
        "hours_train": round(hours(train_rows), 4),
        "hours_validation": round(hours(dev_rows), 4),
        "speaker_map_kin": SPEAKER_MAP_KIN,
        "speaker_map_dav": dav_spk_map,
        "n_speakers": 3 + len(dav_spk_map),
        "sample_rate": 16000,
        **meta_extra,
    }
    (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"MMS bilingual HF → {out_dir}")
    print(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kin_manifest_dir",
        type=Path,
        default=Path(
            "/project/community/rmwisene/tts_data_cleaning/"
            "kinyarwanda_tts_mbaza_afrispeak/manifests"
        ),
    )
    ap.add_argument(
        "--dav_manifest_dir",
        type=Path,
        default=Path(
            "/project/community/rmwisene/tts_data_cleaning/"
            "kidawida_tts_prepared/manifests"
        ),
    )
    ap.add_argument(
        "--mode",
        choices=("balanced", "full"),
        default="balanced",
        help="balanced≈match Kin train hours; full=all Dav train",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--target_dav_hours",
        type=float,
        default=0.0,
        help="Override Dav hours for balanced (0 = use Kin train hours)",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Default: bilingual_mms_hf_balanced or bilingual_mms_hf_full",
    )
    ap.add_argument("--limit", type=int, default=0, help="Smoke: cap each source train")
    args = ap.parse_args()

    cleaning = Path("/project/community/rmwisene/tts_data_cleaning")
    if args.out_dir is None:
        suffix = "balanced" if args.mode == "balanced" else "full"
        args.out_dir = cleaning / f"bilingual_mms_hf_{suffix}"

    kin_train = read_manifest(args.kin_manifest_dir / "train.csv")
    kin_dev = read_manifest(args.kin_manifest_dir / "dev.csv")
    dav_train = read_manifest(args.dav_manifest_dir / "train.csv")
    dav_dev = read_manifest(args.dav_manifest_dir / "dev.csv")

    if args.limit:
        kin_train = kin_train[: args.limit]
        dav_train = dav_train[: args.limit]
        kin_dev = kin_dev[: max(1, args.limit // 10)]
        dav_dev = dav_dev[: max(1, args.limit // 10)]

    kin_h = hours(kin_train)
    dav_h_all = hours(dav_train)
    print(f"Kin train: {len(kin_train)} clips, {kin_h:.2f} h")
    print(f"Dav train: {len(dav_train)} clips, {dav_h_all:.2f} h")

    if args.mode == "balanced":
        target = args.target_dav_hours if args.target_dav_hours > 0 else kin_h
        dav_train_use = sample_by_hours(dav_train, target, args.seed)
        print(
            f"Dav subset (seed={args.seed}, target={target:.2f} h): "
            f"{len(dav_train_use)} clips, {hours(dav_train_use):.2f} h"
        )
    else:
        dav_train_use = dav_train

    # Tag sources if missing
    for r in dav_train_use + dav_dev:
        if not (r.get("source") or "").strip():
            r["source"] = "kidawida_cv27"

    train = kin_train + dav_train_use
    # shuffle train for mixing
    rng = random.Random(args.seed)
    rng.shuffle(train)
    dev = kin_dev + dav_dev

    dav_spk_map: dict[str, int] = {}
    # Pre-register so meta is complete before export assigns
    for r in train + dev:
        speaker_id_for(r, dav_spk_map)

    export_mms_hf(
        train,
        dev,
        args.out_dir,
        dav_spk_map,
        meta_extra={
            "mode": args.mode,
            "seed": args.seed,
            "kin_train_hours": round(kin_h, 4),
            "dav_train_hours_used": round(hours(dav_train_use), 4),
            "dav_train_hours_available": round(dav_h_all, 4),
            "n_kin_train": len(kin_train),
            "n_dav_train": len(dav_train_use),
            "n_kin_dev": len(kin_dev),
            "n_dav_dev": len(dav_dev),
        },
    )


if __name__ == "__main__":
    main()
