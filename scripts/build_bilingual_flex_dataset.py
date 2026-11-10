#!/usr/bin/env python3
"""Build Flex PSV (+ 24 kHz wavs) for bilingual Kin + Kidaw’ida (E4/E6).

Kinya-Flex has 3 speaker slots (0/1/2). Kin sources keep their map;
Dav speakers are hashed into {0,1,2} (no new embeddings).

  # E4 balanced (~Kin hours of Dav)
  python scripts/build_bilingual_flex_dataset.py --mode balanced --seed 42

  # E6 full Dav
  python scripts/build_bilingual_flex_dataset.py --mode full
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from math import gcd
from pathlib import Path

import numpy as np

SPEAKER_MAP_KIN = {
    "afrispeak_female": 0,
    "mbaza_tts": 1,
    "afrispeak_male": 2,
}


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def hours(rows: list[dict]) -> float:
    return sum(float(r.get("duration") or 0.0) for r in rows) / 3600.0


def sample_by_hours(rows: list[dict], target_h: float, seed: int) -> list[dict]:
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


def flex_speaker_id(row: dict) -> int:
    src = (row.get("source") or "").strip()
    if src in SPEAKER_MAP_KIN:
        return SPEAKER_MAP_KIN[src]
    # Dav → existing Flex slots only
    key = (row.get("speaker_id") or src or "dav").encode("utf-8")
    return int(hashlib.md5(key).hexdigest(), 16) % 3


def resample_to_24k(src: Path, dst: Path) -> None:
    import soundfile as sf

    if dst.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    arr, sr = sf.read(str(src), always_2d=False)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    if int(sr) != 24000:
        try:
            import librosa

            arr = librosa.resample(arr, orig_sr=int(sr), target_sr=24000)
        except Exception:
            from scipy.signal import resample_poly

            g = gcd(24000, int(sr))
            arr = resample_poly(arr, 24000 // g, int(sr) // g).astype(np.float32)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1.0:
        arr = arr / peak
    sf.write(str(dst), arr, 24000, subtype="PCM_16")


def write_psv(rows: list[dict], out_dir: Path, psv_name: str) -> int:
    lines: list[str] = []
    for i, r in enumerate(rows, 1):
        src = Path(r["audio_path"])
        text = (r.get("text") or "").replace("|", " ").strip()
        if not text or not src.is_file():
            continue
        utt = r.get("utt_id") or src.stem
        rel = f"wav24k/{utt}.wav"
        resample_to_24k(src, out_dir / rel)
        spk = flex_speaker_id(r)
        lines.append(f"{rel}|{spk}|{text}")
        if i % 1000 == 0:
            print(f"  {psv_name}: {i}/{len(rows)}")
    psv = out_dir / psv_name
    psv.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"Flex {psv_name}: {len(lines)} → {psv}")
    return len(lines)


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
    ap.add_argument("--mode", choices=("balanced", "full"), default="balanced")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_dav_hours", type=float, default=0.0)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cleaning = Path("/project/community/rmwisene/tts_data_cleaning")
    if args.out_dir is None:
        suffix = "balanced" if args.mode == "balanced" else "full"
        args.out_dir = cleaning / f"bilingual_flex_data_{suffix}"

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
    print(f"Kin train: {len(kin_train)} clips, {kin_h:.2f} h")
    print(f"Dav train: {len(dav_train)} clips, {hours(dav_train):.2f} h")

    if args.mode == "balanced":
        target = args.target_dav_hours if args.target_dav_hours > 0 else kin_h
        dav_train_use = sample_by_hours(dav_train, target, args.seed)
        print(
            f"Dav subset (seed={args.seed}, target={target:.2f} h): "
            f"{len(dav_train_use)} clips, {hours(dav_train_use):.2f} h"
        )
    else:
        dav_train_use = dav_train

    for r in dav_train_use + dav_dev:
        if not (r.get("source") or "").strip():
            r["source"] = "kidawida_asr_expanded"

    train = kin_train + dav_train_use
    rng = random.Random(args.seed)
    rng.shuffle(train)
    dev = kin_dev + dav_dev

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_tr = write_psv(train, args.out_dir, "tts_train_data.psv")
    n_dv = write_psv(dev, args.out_dir, "tts_dev_data.psv")
    meta = {
        "mode": args.mode,
        "seed": args.seed,
        "n_train_psv": n_tr,
        "n_dev_psv": n_dv,
        "kin_train_hours": round(kin_h, 4),
        "dav_train_hours_used": round(hours(dav_train_use), 4),
        "dav_train_hours_available": round(hours(dav_train), 4),
        "speaker_map_kin": SPEAKER_MAP_KIN,
        "dav_speaker_policy": "md5(client_id) % 3  (Flex has 3 slots only)",
        "sample_rate": 24000,
    }
    (args.out_dir / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Flex bilingual data → {args.out_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
