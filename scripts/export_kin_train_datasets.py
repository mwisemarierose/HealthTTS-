#!/usr/bin/env python3
"""Export Kin merged manifests for E1 (MMS) and E2 (Flex) training.

Reads:
  …/kinyarwanda_tts_mbaza_afrispeak/manifests/{train,dev}.csv

Writes:
  MMS: HF DatasetDict on disk (16 kHz audio paths + text + speaker_id)
  Flex: PSV list + optional 24 kHz wav copies (DeepKIN expects 24 kHz)

  python scripts/export_kin_train_datasets.py
  python scripts/export_kin_train_datasets.py --skip_flex_resample  # PSV only, paths stay 16 kHz (will fail Flex SR check)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SPEAKER_MAP = {
    "afrispeak_female": 0,  # Flex: Female1
    "mbaza_tts": 1,  # Flex: Female2 / other
    "afrispeak_male": 2,  # Flex: Male
}


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def export_mms_hf(train_rows: list[dict], dev_rows: list[dict], out_dir: Path) -> None:
    from datasets import Audio, Dataset, DatasetDict, Features, Value

    def to_ds(rows: list[dict]) -> Dataset:
        return Dataset.from_dict(
            {
                "audio": [r["audio_path"] for r in rows],
                "text": [r["text"] for r in rows],
                "speaker_id": [int(SPEAKER_MAP.get(r.get("source", ""), 0)) for r in rows],
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
        "speaker_map": SPEAKER_MAP,
        "sample_rate": 16000,
    }
    (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"MMS HF dataset → {out_dir}  train={len(train_rows)}  val={len(dev_rows)}")


def export_flex_psv(
    train_rows: list[dict],
    dev_rows: list[dict],
    out_dir: Path,
    resample_24k: bool,
) -> None:
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    audio_out = out_dir / "wav24k"
    if resample_24k:
        audio_out.mkdir(parents=True, exist_ok=True)

    def convert_split(rows: list[dict], psv_name: str) -> None:
        lines: list[str] = []
        for i, r in enumerate(rows, 1):
            src = Path(r["audio_path"])
            spk = SPEAKER_MAP.get(r.get("source", ""), 0)
            text = (r.get("text") or "").replace("|", " ").strip()
            if not text or not src.is_file():
                continue
            if resample_24k:
                rel = f"wav24k/{r.get('utt_id', src.stem)}.wav"
                dst = out_dir / rel
                if not dst.is_file():
                    arr, sr = sf.read(str(src), always_2d=False)
                    arr = np.asarray(arr, dtype=np.float32)
                    if arr.ndim > 1:
                        arr = arr.mean(axis=-1)
                    if int(sr) != 24000:
                        try:
                            import librosa

                            arr = librosa.resample(arr, orig_sr=int(sr), target_sr=24000)
                        except Exception:
                            from math import gcd
                            from scipy.signal import resample_poly

                            g = gcd(24000, int(sr))
                            arr = resample_poly(arr, 24000 // g, int(sr) // g).astype(np.float32)
                    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
                    if peak > 1.0:
                        arr = arr / peak
                    sf.write(str(dst), arr, 24000, subtype="PCM_16")
                # path relative to out_dir (tts_data_dir)
                lines.append(f"{rel}|{spk}|{text}")
            else:
                lines.append(f"{src.resolve()}|{spk}|{text}")
            if i % 1000 == 0:
                print(f"  {psv_name}: {i}/{len(rows)}")
        psv = out_dir / psv_name
        psv.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"Flex {psv_name}: {len(lines)} → {psv}")

    convert_split(train_rows, "tts_train_data.psv")
    convert_split(dev_rows, "tts_dev_data.psv")
    (out_dir / "export_meta.json").write_text(
        json.dumps(
            {
                "n_train_psv": "see file",
                "speaker_map": SPEAKER_MAP,
                "sample_rate": 24000 if resample_24k else 16000,
                "resample_24k": resample_24k,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Flex data dir → {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest_dir",
        type=Path,
        default=Path(
            "/project/community/rmwisene/tts_data_cleaning/"
            "kinyarwanda_tts_mbaza_afrispeak/manifests"
        ),
    )
    ap.add_argument(
        "--mms_out",
        type=Path,
        default=Path("/project/community/rmwisene/tts_data_cleaning/kin_mms_hf"),
    )
    ap.add_argument(
        "--flex_out",
        type=Path,
        default=Path("/project/community/rmwisene/tts_data_cleaning/kin_flex_data"),
    )
    ap.add_argument("--skip_mms", action="store_true")
    ap.add_argument("--skip_flex", action="store_true")
    ap.add_argument(
        "--skip_flex_resample",
        action="store_true",
        help="Do not resample to 24 kHz (Flex train will fail SR check)",
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    train = read_manifest(args.manifest_dir / "train.csv")
    dev = read_manifest(args.manifest_dir / "dev.csv")
    if args.limit:
        train, dev = train[: args.limit], dev[: max(1, args.limit // 10)]
    print(f"Loaded train={len(train)}  dev={len(dev)}")

    if not args.skip_mms:
        export_mms_hf(train, dev, args.mms_out)
    if not args.skip_flex:
        export_flex_psv(train, dev, args.flex_out, resample_24k=not args.skip_flex_resample)


if __name__ == "__main__":
    main()
