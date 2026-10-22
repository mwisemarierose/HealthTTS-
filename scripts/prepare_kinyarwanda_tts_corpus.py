#!/usr/bin/env python3
"""Prepare merged Kinyarwanda TTS corpus: Pastor (STOI≥0.90) + mbaza.

Order:
  1) Load Pastor  2) Filter/clean  3) Load mbaza  4) Clean/standardize
  5) Merge  6) QC (dupes / bad text / audio)  7) train/dev/test split
  8) Same manifests for Kinya-Flex and MMS-TTS-Kin

Does NOT modify raw datasets. Audio copies written under output/audio/.

  python scripts/prepare_kinyarwanda_tts_corpus.py
  python scripts/prepare_kinyarwanda_tts_corpus.py --limit 50   # smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install pyyaml") from e

try:
    import soundfile as sf
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install soundfile") from e

QUOTE_MAP = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201A": "'",
    "\u201B": "'",
    "\u2032": "'",
    "\u201C": '"',
    "\u201D": '"',
    "\u201E": '"',
    "\u2033": '"',
    "\u00AB": '"',
    "\u00BB": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u2026": "...",
    "\u00A0": " ",
    "\u2009": " ",
    "\u200A": " ",
    "\u202F": " ",
}
WS_RE = re.compile(r"\s+")
LINE_QUOTED = re.compile(r'^\s*(\S+)\s+"(.*)"\s*$')
SCHEMA = ("utt_id", "audio_path", "text", "speaker_id", "source", "duration")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def mild_text(text: str, cfg: dict) -> str:
    """Preserve case + useful punctuation; no ASR-style wipe."""
    t = text or ""
    if cfg.get("normalize_unicode_nfc", True):
        t = unicodedata.normalize("NFC", t)
    t = "".join(QUOTE_MAP.get(ch, ch) for ch in t)
    if cfg.get("collapse_whitespace", True):
        t = WS_RE.sub(" ", t).strip()
    else:
        t = t.strip()
    if cfg.get("lowercase", False):
        t = t.lower()
    return t


def resample_mono(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=-1) if x.shape[-1] <= 8 else x.mean(axis=0)
    x = np.squeeze(x)
    if sr_in == sr_out:
        return x
    try:
        import librosa

        return librosa.resample(x, orig_sr=sr_in, target_sr=sr_out)
    except Exception:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr_out, sr_in)
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sr: int) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(audio, dtype=np.float32)
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak > 1.0:
        x = x / peak
    sf.write(str(path), x, sr, subtype="PCM_16")
    return float(len(x) / sr) if sr else 0.0


def audio_duration(path: Path) -> float | None:
    try:
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def load_pastor_dataset(local_dir: Path, hf_id: str):
    """Load Pastor split with audio decode disabled (filter STOI first)."""
    from datasets import Audio, load_dataset

    local_dir = Path(local_dir)
    if local_dir.is_dir() and any(local_dir.rglob("*.parquet")):
        data_dir = local_dir / "data"
        if data_dir.is_dir():
            ds = load_dataset("parquet", data_dir=str(data_dir))
        else:
            ds = load_dataset("parquet", data_files=str(local_dir / "**" / "*.parquet"))
    elif local_dir.is_dir() and (local_dir / "dataset_info.json").is_file():
        from datasets import load_from_disk

        ds = load_from_disk(str(local_dir))
    else:
        ds = load_dataset(hf_id)

    split_name = "train" if "train" in ds else list(ds.keys())[0]
    split = ds[split_name]
    # Do NOT decode audio while filtering — decoding on full iterate fails / is slow
    if "audio" in split.column_names:
        split = split.cast_column("audio", Audio(sampling_rate=16000, decode=False))
    return split


def decode_hf_audio(audio_obj, target_sr: int = 16000):
    """Decode HF Audio(decode=False) dict → (waveform, sr) or (None, None)."""
    import io

    if audio_obj is None:
        return None, None
    if isinstance(audio_obj, dict):
        if audio_obj.get("array") is not None:
            return np.asarray(audio_obj["array"], dtype=np.float32), int(
                audio_obj.get("sampling_rate") or target_sr
            )
        raw = audio_obj.get("bytes")
        if raw:
            arr, sr = sf.read(io.BytesIO(raw), always_2d=False)
            return np.asarray(arr, dtype=np.float32), int(sr)
        path = audio_obj.get("path")
        if path and Path(path).is_file():
            arr, sr = sf.read(str(path), always_2d=False)
            return np.asarray(arr, dtype=np.float32), int(sr)
    return None, None


def pastor_stats(rows: list[dict]) -> dict:
    durs = np.array([r["duration"] for r in rows], dtype=float)
    texts = [r["text"] for r in rows]
    empty = sum(1 for t in texts if not (t or "").strip())
    stripped = [str(t).strip() for t in texts if t is not None]
    n_dup = len(stripped) - len(set(stripped))
    lens = np.array([len(t) for t in stripped], dtype=float) if stripped else np.array([0.0])
    words = np.array(
        [len(t.split()) for t in stripped], dtype=float
    ) if stripped else np.array([0.0])
    return {
        "n_clips": len(rows),
        "total_duration_hours": round(float(durs.sum()) / 3600.0, 4) if len(durs) else 0.0,
        "duration_s_mean": round(float(durs.mean()), 4) if len(durs) else None,
        "duration_s_median": round(float(np.median(durs)), 4) if len(durs) else None,
        "duration_s_min": round(float(durs.min()), 4) if len(durs) else None,
        "duration_s_max": round(float(durs.max()), 4) if len(durs) else None,
        "empty_transcripts": empty,
        "duplicate_transcripts": n_dup,
        "text_chars_mean": round(float(lens.mean()), 2),
        "text_chars_median": round(float(np.median(lens)), 2),
        "text_words_mean": round(float(words.mean()), 2),
        "text_words_median": round(float(np.median(words)), 2),
    }


def parse_mbaza_metadata(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith(("license", "http")):
                continue
            m = LINE_QUOTED.match(line)
            if not m:
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                utt_id, text = parts[0], parts[1].strip().strip('"')
            else:
                utt_id, text = m.group(1), m.group(2)
            if not text:
                continue
            rows.append({"id": utt_id.strip(), "original_text": text})
    return rows


def index_mbaza_audio(audio_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for p in audio_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}:
            index[p.name] = p
            index[p.stem] = p
    return index


def resolve_mbaza_audio(utt_id: str, index: dict[str, Path], audio_dir: Path) -> Path | None:
    stems = [utt_id]
    if re.match(r"^TTS_\d", utt_id, flags=re.I):
        stems.append(re.sub(r"^TTS_", "TTS ", utt_id, count=1, flags=re.I))
    for stem in stems:
        for key in (stem, f"{stem}.wav", f"{stem}.flac", f"{stem}.mp3"):
            if key in index:
                return index[key]
        for ext in (".wav", ".flac", ".mp3"):
            p = audio_dir / f"{stem}{ext}"
            if p.is_file():
                return p
    return None


def reject(rows: list[dict], reason: str, rejected: list[dict], extra: dict | None = None) -> None:
    for r in rows:
        item = {
            "utt_id": r.get("utt_id", ""),
            "source": r.get("source", ""),
            "reason": reason,
            "text": (r.get("text") or "")[:200],
            "audio_path": r.get("audio_path", ""),
        }
        if extra:
            item.update(extra)
        rejected.append(item)


def stratified_split(
    rows: list[dict], seed: int, train_r: float, dev_r: float, test_r: float, stratify: bool
) -> dict[str, list[dict]]:
    assert abs(train_r + dev_r + test_r - 1.0) < 1e-6
    rng = np.random.default_rng(seed)

    def split_group(group: list[dict]) -> dict[str, list[dict]]:
        idx = np.arange(len(group))
        rng.shuffle(idx)
        n = len(group)
        n_test = max(1, int(round(n * test_r))) if n >= 20 else max(0, int(n * test_r))
        n_dev = max(1, int(round(n * dev_r))) if n >= 20 else max(0, int(n * dev_r))
        if n_test + n_dev >= n:
            n_test = max(0, n // 10)
            n_dev = max(0, n // 10)
        n_train = n - n_dev - n_test
        order = idx.tolist()
        return {
            "train": [group[i] for i in order[:n_train]],
            "dev": [group[i] for i in order[n_train : n_train + n_dev]],
            "test": [group[i] for i in order[n_train + n_dev :]],
        }

    if stratify:
        by_src: dict[str, list[dict]] = {}
        for r in rows:
            by_src.setdefault(r["source"], []).append(r)
        out = {"train": [], "dev": [], "test": []}
        for src, group in by_src.items():
            part = split_group(group)
            for k in out:
                out[k].extend(part[k])
        for k in out:
            rng.shuffle(out[k])
        return out
    return split_group(rows)


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(SCHEMA))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SCHEMA})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "kinyarwanda_tts_merge.yaml",
    )
    ap.add_argument("--limit", type=int, default=0, help="Limit per source (smoke)")
    ap.add_argument(
        "--skip_audio_export",
        action="store_true",
        help="Keep original audio paths (still require readable audio)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["output"]["dir"])
    audio_out = out_dir / "audio"
    man_dir = out_dir / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    clean_cfg = cfg["cleaning"]
    sr_out = int(cfg["audio"]["sample_rate"])
    rejected: list[dict] = []

    # ------------------------------------------------------------------
    # 1–2) Pastor: load, filter STOI, stats, review sample, export
    # ------------------------------------------------------------------
    print("=== 1/2 Pastor Uwambaje ===")
    pastor_cfg = cfg["pastor"]
    split = load_pastor_dataset(Path(pastor_cfg["local_dir"]), pastor_cfg["hf_id"])
    print(f"  loaded rows: {len(split)}  columns: {split.column_names}")

    min_stoi = float(pastor_cfg["min_squim_stoi"])
    # Column access avoids decoding audio for every row
    stoi_col = split["squim_stoi"]
    text_col = split["text"]
    dur_col = split["duration"]
    n_below = 0
    pastor_kept_raw: list[dict] = []
    for i in range(len(split)):
        stoi = float(stoi_col[i] or 0.0)
        text_raw = text_col[i] or ""
        dur = float(dur_col[i] or 0.0)
        if stoi < min_stoi:
            n_below += 1
            continue
        pastor_kept_raw.append(
            {
                "idx": i,
                "utt_id": f"pastor_{i:06d}",
                "text_raw": text_raw,
                "duration_meta": dur,
                "squim_stoi": stoi,
                "speaker_id": pastor_cfg["speaker_id"],
                "source": pastor_cfg["source"],
            }
        )
        if args.limit and len(pastor_kept_raw) >= args.limit:
            break

    print(f"  after STOI>={min_stoi}: {len(pastor_kept_raw)}  (excluded below threshold: {n_below})")
    rejected.append(
        {
            "utt_id": "AGGREGATE",
            "source": pastor_cfg["source"],
            "reason": f"stoi_below_{min_stoi}",
            "text": f"count={n_below}",
            "audio_path": "",
            "squim_stoi": "",
        }
    )

    # Stats on filtered set (pre-export)
    stats_rows = [
        {
            "duration": r["duration_meta"],
            "text": mild_text(str(r["text_raw"]), clean_cfg),
        }
        for r in pastor_kept_raw
    ]
    pstats = pastor_stats(stats_rows)
    pstats["min_squim_stoi"] = min_stoi
    (out_dir / "pastor_stoi90_stats.json").write_text(
        json.dumps(pstats, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  Pastor stats: {pstats['n_clips']} clips, {pstats['total_duration_hours']} h")

    # Manual review sample
    review_n = int(cfg["review"]["n_pastor_samples"])
    review_seed = int(cfg["review"]["seed"])
    rng = np.random.default_rng(review_seed)
    review_idx = list(range(len(pastor_kept_raw)))
    rng.shuffle(review_idx)
    review_idx = review_idx[: min(review_n, len(review_idx))]
    review_path = out_dir / "pastor_manual_review_sample.csv"
    with review_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "utt_id",
                "text",
                "duration",
                "squim_stoi",
                "speaker_id",
                "source",
                "note",
            ],
        )
        w.writeheader()
        for j in review_idx:
            r = pastor_kept_raw[j]
            w.writerow(
                {
                    "utt_id": r["utt_id"],
                    "text": mild_text(str(r["text_raw"]), clean_cfg),
                    "duration": r["duration_meta"],
                    "squim_stoi": r["squim_stoi"],
                    "speaker_id": r["speaker_id"],
                    "source": r["source"],
                    "note": "Gemini Flash transcript — listen and verify",
                }
            )
    print(f"  review sample: {len(review_idx)} → {review_path}")

    pastor_rows: list[dict] = []
    print(f"  exporting Pastor audio → {audio_out / 'pastor'} …")
    for n_done, r in enumerate(pastor_kept_raw, 1):
        text = mild_text(str(r["text_raw"]), clean_cfg)
        if clean_cfg.get("drop_empty_text", True) and not text:
            reject([{"utt_id": r["utt_id"], "source": r["source"], "text": "", "audio_path": ""}], "empty_text", rejected)
            continue
        if len(text) < int(clean_cfg["min_chars"]) or len(text) > int(clean_cfg["max_chars"]):
            reject(
                [{"utt_id": r["utt_id"], "source": r["source"], "text": text, "audio_path": ""}],
                "text_length",
                rejected,
            )
            continue

        # Decode audio only for kept rows
        try:
            audio_obj = split[r["idx"]]["audio"]
        except Exception as e:
            reject(
                [
                    {
                        "utt_id": r["utt_id"],
                        "source": r["source"],
                        "text": text,
                        "audio_path": "",
                    }
                ],
                f"audio_row_error:{type(e).__name__}",
                rejected,
            )
            continue

        arr, sr_in = decode_hf_audio(audio_obj, target_sr=sr_out)
        if arr is None:
            reject(
                [{"utt_id": r["utt_id"], "source": r["source"], "text": text, "audio_path": ""}],
                "missing_or_undecodable_audio",
                rejected,
            )
            continue

        if args.skip_audio_export and isinstance(audio_obj, dict) and audio_obj.get("path"):
            out_path = Path(audio_obj["path"])
            dur = audio_duration(out_path) or float(r["duration_meta"])
        else:
            arr = resample_mono(arr, int(sr_in or sr_out), sr_out)
            out_path = audio_out / "pastor" / f"{r['utt_id']}.wav"
            dur = write_wav(out_path, arr, sr_out)

        if dur < float(clean_cfg["min_duration_s"]) or dur > float(clean_cfg["max_duration_s"]):
            reject(
                [
                    {
                        "utt_id": r["utt_id"],
                        "source": r["source"],
                        "text": text,
                        "audio_path": str(out_path),
                    }
                ],
                "duration_out_of_range",
                rejected,
            )
            continue

        pastor_rows.append(
            {
                "utt_id": r["utt_id"],
                "audio_path": str(out_path.resolve()),
                "text": text,
                "speaker_id": r["speaker_id"],
                "source": r["source"],
                "duration": round(dur, 3),
            }
        )
        if n_done % 500 == 0 or n_done == len(pastor_kept_raw):
            print(f"    pastor export {n_done}/{len(pastor_kept_raw)}  kept={len(pastor_rows)}")

    print(f"  Pastor usable: {len(pastor_rows)}")

    # ------------------------------------------------------------------
    # 3–4) mbaza: load, mild text (no lowercase), standardize audio
    # ------------------------------------------------------------------
    print("=== 3/4 mbaza TTS ===")
    mbaza_cfg = cfg["mbaza"]
    root = Path(mbaza_cfg["data_root"])
    meta = root / mbaza_cfg["metadata"]
    audio_dir = root / mbaza_cfg["audio_dir"]
    mbaza_meta = parse_mbaza_metadata(meta)
    if args.limit:
        mbaza_meta = mbaza_meta[: args.limit]
    index = index_mbaza_audio(audio_dir)
    print(f"  metadata: {len(mbaza_meta)}  audio files indexed: {len(set(index.values()))}")

    mbaza_rows: list[dict] = []
    for item in mbaza_meta:
        utt_id = item["id"]
        text = mild_text(item["original_text"], clean_cfg)
        src_audio = resolve_mbaza_audio(utt_id, index, audio_dir)
        safe_id = re.sub(r"[^\w.\-]+", "_", utt_id)

        if not text:
            reject(
                [{"utt_id": f"mbaza_{safe_id}", "source": mbaza_cfg["source"], "text": "", "audio_path": ""}],
                "empty_text",
                rejected,
            )
            continue
        if len(text) < int(clean_cfg["min_chars"]) or len(text) > int(clean_cfg["max_chars"]):
            reject(
                [
                    {
                        "utt_id": f"mbaza_{safe_id}",
                        "source": mbaza_cfg["source"],
                        "text": text,
                        "audio_path": "",
                    }
                ],
                "text_length",
                rejected,
            )
            continue
        if src_audio is None:
            reject(
                [
                    {
                        "utt_id": f"mbaza_{safe_id}",
                        "source": mbaza_cfg["source"],
                        "text": text,
                        "audio_path": "",
                    }
                ],
                "missing_audio",
                rejected,
            )
            continue

        try:
            arr, sr_in = sf.read(str(src_audio), always_2d=False)
        except Exception:
            reject(
                [
                    {
                        "utt_id": f"mbaza_{safe_id}",
                        "source": mbaza_cfg["source"],
                        "text": text,
                        "audio_path": str(src_audio),
                    }
                ],
                "unreadable_audio",
                rejected,
            )
            continue

        if args.skip_audio_export and int(sr_in) == sr_out:
            out_path = src_audio
            dur = audio_duration(out_path) or 0.0
        else:
            arr = resample_mono(np.asarray(arr, dtype=np.float32), int(sr_in), sr_out)
            out_path = audio_out / "mbaza" / f"{safe_id}.wav"
            dur = write_wav(out_path, arr, sr_out)

        if dur < float(clean_cfg["min_duration_s"]) or dur > float(clean_cfg["max_duration_s"]):
            reject(
                [
                    {
                        "utt_id": f"mbaza_{safe_id}",
                        "source": mbaza_cfg["source"],
                        "text": text,
                        "audio_path": str(out_path),
                    }
                ],
                "duration_out_of_range",
                rejected,
            )
            continue

        mbaza_rows.append(
            {
                "utt_id": f"mbaza_{safe_id}",
                "audio_path": str(Path(out_path).resolve()),
                "text": text,
                "speaker_id": mbaza_cfg["speaker_id"],
                "source": mbaza_cfg["source"],
                "duration": round(float(dur), 3),
            }
        )

    print(f"  mbaza usable: {len(mbaza_rows)}")

    # ------------------------------------------------------------------
    # 5–6) Merge + pool QC
    # ------------------------------------------------------------------
    print("=== 5/6 Merge + QC ===")
    pool = pastor_rows + mbaza_rows
    print(f"  pool before dedupe: {len(pool)}")

    if clean_cfg.get("drop_duplicate_text", True):
        seen: set[str] = set()
        deduped = []
        for r in pool:
            key = mild_text(r["text"], clean_cfg)
            # case-sensitive key (we preserve case)
            if key in seen:
                reject([r], "duplicate_text", rejected)
                continue
            seen.add(key)
            deduped.append(r)
        pool = deduped
        print(f"  after text dedupe: {len(pool)}")

    # ------------------------------------------------------------------
    # 7) Split
    # ------------------------------------------------------------------
    print("=== 7 Split ===")
    sp = cfg["split"]
    splits = stratified_split(
        pool,
        seed=int(sp["seed"]),
        train_r=float(sp["train_ratio"]),
        dev_r=float(sp["dev_ratio"]),
        test_r=float(sp["test_ratio"]),
        stratify=bool(sp.get("stratify_by_source", True)),
    )
    for name, rows in splits.items():
        write_manifest(man_dir / f"{name}.csv", rows)
        hours = sum(r["duration"] for r in rows) / 3600.0
        print(f"  {name}: {len(rows)} clips, {hours:.3f} h")

    write_manifest(man_dir / "all.csv", pool)

    # Rejected
    rej_path = out_dir / "rejected.csv"
    with rej_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["utt_id", "source", "reason", "text", "audio_path", "squim_stoi"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rejected:
            w.writerow(r)

    # Dataset statistics
    def src_stats(rows: list[dict]) -> dict:
        by = {}
        for r in rows:
            by.setdefault(r["source"], {"n": 0, "duration_s": 0.0})
            by[r["source"]]["n"] += 1
            by[r["source"]]["duration_s"] += float(r["duration"])
        for k, v in by.items():
            v["duration_hours"] = round(v["duration_s"] / 3600.0, 4)
        return by

    all_stats = {
        "pastor_filtered": pstats,
        "n_pool": len(pool),
        "pool_hours": round(sum(r["duration"] for r in pool) / 3600.0, 4),
        "by_source": src_stats(pool),
        "splits": {
            k: {
                "n": len(v),
                "hours": round(sum(r["duration"] for r in v) / 3600.0, 4),
                "by_source": src_stats(v),
            }
            for k, v in splits.items()
        },
        "n_rejected": len(rejected),
        "reject_reasons": dict(Counter(r["reason"] for r in rejected)),
        "audio": cfg["audio"],
        "cleaning": clean_cfg,
        "note": (
            "Raw Pastor/mbaza datasets were not modified. "
            "Use manifests/train.csv|dev.csv|test.csv for both Flex and MMS."
        ),
    }
    (out_dir / "dataset_statistics.json").write_text(
        json.dumps(all_stats, indent=2) + "\n", encoding="utf-8"
    )

    split_config = {
        "seed": sp["seed"],
        "train_ratio": sp["train_ratio"],
        "dev_ratio": sp["dev_ratio"],
        "test_ratio": sp["test_ratio"],
        "stratify_by_source": sp.get("stratify_by_source", True),
        "pastor_min_squim_stoi": min_stoi,
        "sample_rate": sr_out,
        "lowercase": clean_cfg.get("lowercase", False),
        "output_dir": str(out_dir),
        "manifests": {
            "train": str(man_dir / "train.csv"),
            "dev": str(man_dir / "dev.csv"),
            "test": str(man_dir / "test.csv"),
            "all": str(man_dir / "all.csv"),
        },
    }
    (out_dir / "split_config.json").write_text(
        json.dumps(split_config, indent=2) + "\n", encoding="utf-8"
    )

    report = [
        "Kinyarwanda TTS merged corpus",
        "=============================",
        f"Output: {out_dir}",
        f"Pastor STOI>={min_stoi}: {pstats['n_clips']} clips / {pstats['total_duration_hours']} h (pre-QC filter set)",
        f"Pastor usable after QC: {len(pastor_rows)}",
        f"mbaza usable: {len(mbaza_rows)}",
        f"Merged pool: {len(pool)} clips / {all_stats['pool_hours']} h",
        f"Rejected: {len(rejected)}",
        f"Review sample: {review_path}",
        "",
        "Splits:",
    ]
    for k, v in splits.items():
        report.append(
            f"  {k}: {len(v)} / {all_stats['splits'][k]['hours']} h"
        )
    report += [
        "",
        "Use the same manifests for Kinya-Flex and MMS-TTS-Kin.",
        "Schema: audio_path, text, speaker_id, source, duration (+ utt_id).",
        "",
    ]
    (out_dir / "prepare_report.txt").write_text("\n".join(report), encoding="utf-8")

    print()
    print(report[0])
    for line in report[2:]:
        print(line)
    print(f"Wrote stats → {out_dir / 'dataset_statistics.json'}")
    print(f"Wrote split  → {out_dir / 'split_config.json'}")
    print(f"Wrote rejected → {rej_path}")


if __name__ == "__main__":
    main()
