#!/usr/bin/env python3
"""Prepare Kidaw’ida TTS training pool from CV27 + frozen cleaned transcripts.

Order:
  1) Load frozen cleaned text map
  2) Load official CV train/dev/test (path + sentence + client_id)
  3) Attach cleaned text (keep multi-speaker repeats of the same sentence)
  4) Export 16 kHz mono WAV copies
  5) QC: duration / empty text; identical-audio drop only (not text)
  6) Write manifests with the same schema as the Kin pool

Does NOT modify kidawida_cv27.

  python scripts/prepare_kidawida_tts_corpus.py
  python scripts/prepare_kidawida_tts_corpus.py --limit 50
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from math import gcd
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
SCHEMA = ("utt_id", "audio_path", "text", "speaker_id", "source", "duration", "split", "flagged")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def mild_text(text: str, cfg: dict) -> str:
    t = text or ""
    if cfg.get("normalize_unicode_nfc", True):
        t = nfc(t)
    t = "".join(QUOTE_MAP.get(ch, ch) for ch in t)
    if cfg.get("collapse_whitespace", True):
        t = WS_RE.sub(" ", t).strip()
    else:
        t = t.strip()
    if cfg.get("lowercase", False):
        t = t.lower()
    return t


def detect_col(fieldnames: list[str] | None, keys: tuple[str, ...]) -> str | None:
    if not fieldnames:
        return None
    lower = {(f or "").lstrip("\ufeff").strip().lower(): f for f in fieldnames if f}
    for k in keys:
        if k in lower:
            return lower[k]
    for k in keys:
        for low, orig in lower.items():
            if k in low:
                return orig
    return None


def find_split_table(root: Path, basename: str) -> Path | None:
    """Prefer shallowest match of train.tsv / train.csv etc."""
    name = Path(basename).name
    stem = Path(name).stem
    cands: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name == name or p.name.lower() == name.lower():
            cands.append(p)
        elif p.stem.lower() == stem.lower() and p.suffix.lower() in {".tsv", ".csv"}:
            cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda p: (len(p.parts), str(p)))
    return cands[0]


def index_clips(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    exts = {".mp3", ".wav", ".flac", ".ogg", ".opus"}
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        index[p.name] = p
        index[p.stem] = p
        try:
            rel = str(p.relative_to(root)).replace("\\", "/")
            index[rel] = p
            index[Path(rel).name] = p
        except ValueError:
            pass
    return index


def resolve_clip(path_field: str, index: dict[str, Path], root: Path) -> Path | None:
    if not path_field:
        return None
    raw = path_field.strip()
    for c in (
        raw,
        Path(raw).name,
        Path(raw).stem,
        f"clips/{Path(raw).name}",
        str(Path("clips") / Path(raw).name),
    ):
        if c in index:
            return index[c]
    direct = root / raw
    if direct.is_file():
        return direct
    clip = root / "clips" / Path(raw).name
    if clip.is_file():
        return clip
    return None


def read_audio(path: Path) -> tuple[np.ndarray | None, int | None]:
    try:
        arr, sr = sf.read(str(path), always_2d=False)
        return np.asarray(arr, dtype=np.float32), int(sr)
    except Exception:
        pass
    try:
        import librosa

        arr, sr = librosa.load(str(path), sr=None, mono=True)
        return np.asarray(arr, dtype=np.float32), int(sr)
    except Exception:
        return None, None


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
        from scipy.signal import resample_poly

        g = gcd(sr_out, sr_in)
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sr: int) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        x = x / peak
    sf.write(str(path), x, sr, subtype="PCM_16")
    return float(len(x) / sr) if sr else 0.0


def audio_content_hash(path: Path, chunk: int = 1 << 20) -> str | None:
    try:
        h = hashlib.md5()
        with Path(path).open("rb") as f:
            while True:
                buf = f.read(chunk)
                if not buf:
                    break
                h.update(buf)
        return h.hexdigest()
    except Exception:
        return None


def reject(rows: list[dict], reason: str, rejected: list[dict], extra: dict | None = None) -> None:
    for r in rows:
        item = {
            "utt_id": r.get("utt_id", ""),
            "source": r.get("source", ""),
            "split": r.get("split", ""),
            "reason": reason,
            "text": (r.get("text") or "")[:200],
            "audio_path": r.get("audio_path", ""),
        }
        if extra:
            item.update(extra)
        rejected.append(item)


def load_cleaned_map(path: Path) -> dict[str, dict]:
    """Map NFC(original_text) → cleaned row (first wins if consistent)."""
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            orig = nfc((r.get("original_text") or "").strip())
            if not orig:
                continue
            key = orig
            key_l = orig.lower()
            payload = {
                "cleaned_text": (r.get("cleaned_text") or "").strip(),
                "flagged": (r.get("flagged") or "").strip().lower() in {"yes", "true", "1"},
                "flagged_symbols": r.get("flagged_symbols") or "",
            }
            out.setdefault(key, payload)
            out.setdefault(key_l, payload)
    return out


def load_cv_split(path: Path, split_name: str) -> list[dict]:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:4096]
    try:
        delim = csv.Sniffer().sniff(sample, delimiters="\t,;|").delimiter
    except Exception:
        delim = "\t" if path.suffix.lower() == ".tsv" else ","
    rows: list[dict] = []
    with path.open(encoding="utf-8-sig", newline="", errors="replace") as f:
        reader = csv.DictReader(f, delimiter=delim)
        fields = list(reader.fieldnames or [])
        text_col = detect_col(fields, ("sentence", "text", "transcript", "transcription"))
        path_col = detect_col(fields, ("path", "clip", "filename", "file", "audio", "wav"))
        spk_col = detect_col(fields, ("client_id", "speaker", "speaker_id", "client"))
        if text_col is None or path_col is None:
            raise SystemExit(f"Missing path/sentence columns in {path} (fields={fields})")
        for r in reader:
            text = (r.get(text_col) or "").strip()
            clip = (r.get(path_col) or "").strip()
            if not text or not clip:
                continue
            spk = (r.get(spk_col) or "").strip() if spk_col else ""
            rows.append(
                {
                    "split": split_name,
                    "path": clip,
                    "original_text": text,
                    "speaker_raw": spk,
                }
            )
    return rows


def count_text_duplicates(rows: list[dict]) -> dict:
    counts: Counter[str] = Counter(r["text"] for r in rows)
    return {
        "n_unique_texts": len(counts),
        "n_texts_with_repeats": sum(1 for c in counts.values() if c > 1),
        "n_extra_rows_same_text": sum(c - 1 for c in counts.values() if c > 1),
    }


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(SCHEMA))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SCHEMA})


def safe_utt_id(split: str, clip: str, i: int) -> str:
    stem = re.sub(r"[^\w.\-]+", "_", Path(clip).stem)[:80]
    return f"dav_{split}_{stem}_{i:06d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "kidawida_tts_prepare.yaml",
    )
    ap.add_argument("--limit", type=int, default=0, help="Limit per split (smoke)")
    ap.add_argument(
        "--skip_audio_export",
        action="store_true",
        help="Keep original clip paths (still require readable audio)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["data_root"])
    cleaned_csv = Path(cfg["cleaned_transcripts"])
    out_dir = Path(cfg["output"]["dir"])
    audio_out = out_dir / "audio"
    man_dir = out_dir / "manifests"
    clean_cfg = cfg["cleaning"]
    sr_out = int(cfg["audio"]["sample_rate"])
    source_name = cfg["output"].get("source", "kidawida_cv27")
    default_spk = cfg["output"].get("default_speaker", "dav_spk_unknown")
    rejected: list[dict] = []

    if not root.is_dir():
        raise SystemExit(f"Missing Kidaw’ida root: {root}")
    if not cleaned_csv.is_file():
        raise SystemExit(
            f"Missing frozen cleaned transcripts: {cleaned_csv}\n"
            "Run: python scripts/audit_clean_kidawida_transcripts.py && "
            "python scripts/freeze_cleaned_kidawida.py"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    print("=== Kidaw’ida TTS prepare ===")
    print(f"  root: {root}")
    print(f"  cleaned: {cleaned_csv}")

    cleaned_map = load_cleaned_map(cleaned_csv)
    print(f"  cleaned text keys: {len(cleaned_map)}")

    print("Indexing clips …")
    index = index_clips(root)
    n_audio = len({p.resolve() for p in index.values()})
    print(f"  audio files: {n_audio}")

    split_cfg = cfg.get("split_files") or {
        "train": "train.tsv",
        "dev": "dev.tsv",
        "test": "test.tsv",
    }
    cv_by_split: dict[str, list[dict]] = {}
    for split_name, basename in split_cfg.items():
        table = find_split_table(root, basename)
        if table is None:
            raise SystemExit(f"Could not find {basename} under {root}")
        rows = load_cv_split(table, split_name)
        if args.limit:
            rows = rows[: args.limit]
        cv_by_split[split_name] = rows
        print(f"  {split_name}: {len(rows)} rows ← {table.relative_to(root)}")

    pool: list[dict] = []
    n_no_clean_match = 0
    n_flagged = 0
    global_i = 0

    for split_name, cv_rows in cv_by_split.items():
        print(f"=== Export {split_name} ===")
        for row in cv_rows:
            global_i += 1
            orig = nfc(row["original_text"])
            hit = cleaned_map.get(orig) or cleaned_map.get(orig.lower())
            if hit:
                text = mild_text(hit["cleaned_text"], clean_cfg)
                flagged = bool(hit["flagged"])
            else:
                n_no_clean_match += 1
                # Fallback: same mild rules; lowercase to match frozen Dav policy
                text = mild_text(orig, {**clean_cfg, "lowercase": True})
                flagged = False

            if flagged and not clean_cfg.get("keep_flagged", True):
                reject(
                    [
                        {
                            "utt_id": "",
                            "source": source_name,
                            "split": split_name,
                            "text": text,
                            "audio_path": "",
                        }
                    ],
                    "flagged_excluded",
                    rejected,
                )
                continue

            if clean_cfg.get("drop_empty_text", True) and not text:
                reject(
                    [
                        {
                            "utt_id": "",
                            "source": source_name,
                            "split": split_name,
                            "text": "",
                            "audio_path": "",
                        }
                    ],
                    "empty_text",
                    rejected,
                )
                continue
            if len(text) < int(clean_cfg["min_chars"]) or len(text) > int(clean_cfg["max_chars"]):
                reject(
                    [
                        {
                            "utt_id": "",
                            "source": source_name,
                            "split": split_name,
                            "text": text,
                            "audio_path": "",
                        }
                    ],
                    "text_length",
                    rejected,
                )
                continue

            src_audio = resolve_clip(row["path"], index, root)
            utt_id = safe_utt_id(split_name, row["path"], global_i)
            if src_audio is None:
                reject(
                    [
                        {
                            "utt_id": utt_id,
                            "source": source_name,
                            "split": split_name,
                            "text": text,
                            "audio_path": row["path"],
                        }
                    ],
                    "missing_audio",
                    rejected,
                )
                continue

            arr, sr_in = read_audio(src_audio)
            if arr is None or not sr_in:
                reject(
                    [
                        {
                            "utt_id": utt_id,
                            "source": source_name,
                            "split": split_name,
                            "text": text,
                            "audio_path": str(src_audio),
                        }
                    ],
                    "unreadable_audio",
                    rejected,
                )
                continue

            if args.skip_audio_export and int(sr_in) == sr_out and src_audio.suffix.lower() == ".wav":
                out_path = src_audio
                dur = float(len(arr) / sr_in)
            else:
                arr = resample_mono(arr, int(sr_in), sr_out)
                out_path = audio_out / split_name / f"{utt_id}.wav"
                dur = write_wav(out_path, arr, sr_out)

            if dur < float(clean_cfg["min_duration_s"]) or dur > float(clean_cfg["max_duration_s"]):
                reject(
                    [
                        {
                            "utt_id": utt_id,
                            "source": source_name,
                            "split": split_name,
                            "text": text,
                            "audio_path": str(out_path),
                        }
                    ],
                    "duration_out_of_range",
                    rejected,
                )
                continue

            spk = row["speaker_raw"] or default_spk
            if len(spk) > 64:
                spk = "dav_" + hashlib.md5(spk.encode()).hexdigest()[:16]

            if flagged:
                n_flagged += 1
            pool.append(
                {
                    "utt_id": utt_id,
                    "audio_path": str(Path(out_path).resolve()),
                    "text": text,
                    "speaker_id": spk,
                    "source": source_name,
                    "duration": round(float(dur), 3),
                    "split": split_name,
                    "flagged": "yes" if flagged else "no",
                }
            )

            if global_i % 500 == 0:
                print(f"    processed ~{global_i}  kept={len(pool)}")

    print(f"  pool before audio dedupe: {len(pool)}")
    print(f"  cleaned-map misses (fallback mild+lower): {n_no_clean_match}")
    print(f"  flagged kept: {n_flagged}")

    text_dup_stats = count_text_duplicates(pool)
    print(
        "  text repeats (kept): "
        f"{text_dup_stats['n_texts_with_repeats']} texts × "
        f"+{text_dup_stats['n_extra_rows_same_text']} rows"
    )

    if clean_cfg.get("drop_duplicate_text", False):
        seen: set[str] = set()
        deduped = []
        for r in pool:
            if r["text"] in seen:
                reject([r], "duplicate_text", rejected)
                continue
            seen.add(r["text"])
            deduped.append(r)
        pool = deduped
        print(f"  after text dedupe (enabled): {len(pool)}")

    if clean_cfg.get("drop_duplicate_audio", True):
        seen_hash: set[str] = set()
        deduped = []
        n_unhashed = 0
        for r in pool:
            digest = audio_content_hash(Path(r["audio_path"]))
            if digest is None:
                n_unhashed += 1
                deduped.append(r)
                continue
            if digest in seen_hash:
                reject([r], "duplicate_audio", rejected, extra={"audio_md5": digest})
                continue
            seen_hash.add(digest)
            deduped.append(r)
        n_dropped = len(pool) - len(deduped)
        pool = deduped
        print(
            f"  after audio dedupe: {len(pool)}  "
            f"(dropped={n_dropped}, unreadable_hash={n_unhashed})"
        )

    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for r in pool:
        splits.setdefault(r["split"], []).append(r)

    for name, rows in splits.items():
        write_manifest(man_dir / f"{name}.csv", rows)
        hours = sum(float(r["duration"]) for r in rows) / 3600.0
        print(f"  {name}: {len(rows)} clips, {hours:.3f} h")

    write_manifest(man_dir / "all.csv", pool)

    rej_path = out_dir / "rejected.csv"
    with rej_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["utt_id", "source", "split", "reason", "text", "audio_path", "audio_md5"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rejected:
            w.writerow(r)

    def src_stats(rows: list[dict]) -> dict:
        by: dict[str, dict] = {}
        for r in rows:
            by.setdefault(r["source"], {"n": 0, "duration_s": 0.0})
            by[r["source"]]["n"] += 1
            by[r["source"]]["duration_s"] += float(r["duration"])
        for v in by.values():
            v["duration_hours"] = round(v["duration_s"] / 3600.0, 4)
        return by

    all_stats = {
        "n_pool": len(pool),
        "pool_hours": round(sum(float(r["duration"]) for r in pool) / 3600.0, 4),
        "n_flagged_kept": n_flagged,
        "n_cleaned_map_miss": n_no_clean_match,
        "by_source": src_stats(pool),
        "text_duplicate_report": text_dup_stats,
        "n_speakers": len({r["speaker_id"] for r in pool}),
        "splits": {
            k: {
                "n": len(v),
                "hours": round(sum(float(r["duration"]) for r in v) / 3600.0, 4),
            }
            for k, v in splits.items()
        },
        "n_rejected": len(rejected),
        "reject_reasons": dict(Counter(r["reason"] for r in rejected)),
        "audio": cfg["audio"],
        "cleaning": clean_cfg,
        "cleaned_transcripts": str(cleaned_csv),
        "note": (
            "Official CV train/dev/test preserved. "
            "Duplicate text kept; only identical audio bytes dropped. "
            "Same schema as Kin (+ split, flagged). "
            "Source kidawida_cv27 not modified."
        ),
    }
    (out_dir / "dataset_statistics.json").write_text(
        json.dumps(all_stats, indent=2) + "\n", encoding="utf-8"
    )

    split_config = {
        "kind": "common_voice_official",
        "sample_rate": sr_out,
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
        "Kidaw’ida TTS prepared corpus",
        "=============================",
        f"Output: {out_dir}",
        f"Pool: {len(pool)} clips / {all_stats['pool_hours']} h",
        f"Speakers: {all_stats['n_speakers']}",
        f"Flagged kept: {n_flagged}",
        f"Rejected: {len(rejected)}",
        "",
        "Splits (CV official):",
    ]
    for k in ("train", "dev", "test"):
        if k in all_stats["splits"]:
            report.append(
                f"  {k}: {all_stats['splits'][k]['n']} / "
                f"{all_stats['splits'][k]['hours']} h"
            )
    report += [
        "",
        "Schema: utt_id, audio_path, text, speaker_id, source, duration, split, flagged",
        "Use with Flex / MMS Dav adaptation (init from Kin MMS).",
        "",
    ]
    (out_dir / "prepare_report.txt").write_text("\n".join(report), encoding="utf-8")

    print()
    for line in report:
        print(line)
    print(f"Wrote stats → {out_dir / 'dataset_statistics.json'}")
    print(f"Wrote rejected → {rej_path}")


if __name__ == "__main__":
    main()
