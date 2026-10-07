#!/usr/bin/env python3
"""Clean FLEURS Kinyarwanda for baseline evaluation (test/dev only).

- Extracts audio archives if needed
- Builds cleaned manifests (path, text, duration, split)
- Never includes train in eval outputs

  conda activate healthtts
  python scripts/clean_fleurs_eval.py
  python scripts/clean_fleurs_eval.py --config configs/fleurs_eval.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import tarfile
import unicodedata
from collections import defaultdict
from pathlib import Path

import yaml

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None


WHITESPACE_RE = re.compile(r"\s+")


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def normalize_text(text: str, nfc: bool, collapse_ws: bool) -> str:
    t = text.strip()
    if nfc:
        t = unicodedata.normalize("NFC", t)
    if collapse_ws:
        t = WHITESPACE_RE.sub(" ", t)
    return t


def extract_archives(root: Path, archives: list[str]) -> None:
    for rel in archives:
        arc = root / rel
        if not arc.is_file():
            print(f"[skip extract] missing {arc}")
            continue
        # extract next to archive parent (usually audio/)
        dest = arc.parent
        marker = dest / f".extracted_{arc.name}"
        if marker.is_file():
            print(f"[ok] already extracted {arc.name}")
            continue
        print(f"[extract] {arc} → {dest}")
        with tarfile.open(arc, "r:xz") as tf:
            tf.extractall(path=dest)
        marker.write_text("ok\n", encoding="utf-8")


def index_audio_files(root: Path, search_roots: list[str]) -> dict[str, Path]:
    """Map basename and relative path → absolute Path."""
    index: dict[str, Path] = {}
    exts = {".wav", ".flac", ".mp3", ".ogg", ".opus"}
    for rel in search_roots:
        base = root / rel
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.suffix.lower() not in exts:
                continue
            index[p.name] = p
            try:
                index[str(p.relative_to(root))] = p
            except ValueError:
                pass
            # also key without directories commonly used in TSV
            index[p.stem] = p
    return index


def find_audio(row: dict, index: dict[str, Path], root: Path) -> Path | None:
    candidates = []
    for key in (
        "path",
        "audio",
        "file",
        "filename",
        "audio_path",
        "wav",
        "clip",
    ):
        if key in row and row[key]:
            candidates.append(row[key].strip())
    # FLEURS-style: sometimes id + .wav
    if "id" in row and row["id"]:
        candidates.append(f"{row['id'].strip()}.wav")
        candidates.append(row["id"].strip())

    for c in candidates:
        c = c.replace("\\", "/")
        if c in index:
            return index[c]
        name = Path(c).name
        if name in index:
            return index[name]
        stem = Path(c).stem
        if stem in index:
            return index[stem]
        direct = root / c
        if direct.is_file():
            return direct
        under_audio = root / "audio" / name
        if under_audio.is_file():
            return under_audio
    return None


def pick_text(row: dict) -> str:
    for key in (
        "transcription",
        "raw_transcription",
        "sentence",
        "text",
        "transcript",
    ):
        if key in row and row[key] and str(row[key]).strip():
            return str(row[key])
    # last non-empty field heuristic
    for k, v in row.items():
        if k.lower() in {"id", "path", "file", "filename", "gender", "lang", "speaker"}:
            continue
        if v and isinstance(v, str) and len(v.strip()) > 1 and not v.endswith((".wav", ".flac", ".mp3")):
            return v
    return ""


def duration_seconds(path: Path) -> float | None:
    if sf is None:
        return None
    try:
        info = sf.info(str(path))
        return float(info.duration)
    except Exception:
        return None


def _looks_like_header(fields: list[str]) -> bool:
    joined = " ".join(fields).lower()
    return any(
        k in joined
        for k in ("transcription", "sentence", "path", "filename", "audio", "text")
    ) and not any(f.endswith(".wav") for f in fields)


def read_tsv(path: Path) -> list[dict]:
    """Read FLEURS TSV. mbazaNLP dumps are often headerless:

    - 3 cols: id, file, text
    - 4 cols: id, file, raw_transcription, transcription
    """
    rows: list[dict] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        all_rows = [r for r in reader if r and any(c.strip() for c in r)]
    if not all_rows:
        return []

    start = 0
    fieldnames: list[str] | None = None
    if _looks_like_header(all_rows[0]):
        fieldnames = [c.strip().lower() for c in all_rows[0]]
        start = 1

    for raw in all_rows[start:]:
        cols = [c.strip() for c in raw]
        if fieldnames is not None and len(cols) >= len(fieldnames):
            row = {fieldnames[i]: cols[i] for i in range(len(fieldnames))}
        elif len(cols) >= 4:
            row = {
                "id": cols[0],
                "file": cols[1],
                "raw_transcription": cols[2],
                "transcription": cols[3],
            }
        elif len(cols) == 3:
            row = {"id": cols[0], "file": cols[1], "text": cols[2]}
        elif len(cols) == 2:
            row = {"file": cols[0], "text": cols[1]}
        else:
            continue
        rows.append(row)
    return rows

def clean_split(
    root: Path,
    split: str,
    index: dict[str, Path],
    cfg: dict,
) -> tuple[list[dict], dict]:
    tsv = root / f"{split}.tsv"
    stats = defaultdict(int)
    out_rows: list[dict] = []
    if not tsv.is_file():
        stats["missing_tsv"] = 1
        return out_rows, dict(stats)

    clean_cfg = cfg["cleaning"]
    rows = read_tsv(tsv)
    stats["input_rows"] = len(rows)

    for row in rows:
        text = normalize_text(
            pick_text(row),
            clean_cfg.get("normalize_unicode_nfc", True),
            clean_cfg.get("collapse_whitespace", True),
        )
        if len(text) < clean_cfg.get("min_chars", 1):
            stats["drop_empty_text"] += 1
            continue
        if len(text) > clean_cfg.get("max_chars", 500):
            stats["drop_long_text"] += 1
            continue

        audio = find_audio(row, index, root)
        if audio is None:
            stats["drop_missing_audio"] += 1
            if not clean_cfg.get("drop_missing_audio", True):
                pass
            continue

        dur = duration_seconds(audio)
        if dur is not None:
            if dur < clean_cfg.get("min_duration_s", 0.3):
                stats["drop_short_audio"] += 1
                continue
            if dur > clean_cfg.get("max_duration_s", 30.0):
                stats["drop_long_audio"] += 1
                continue
        else:
            stats["duration_unknown"] += 1

        out_rows.append(
            {
                "id": row.get("id") or audio.stem,
                "split": split,
                "audio_path": str(audio.resolve()),
                "text": text,
                "duration_s": f"{dur:.3f}" if dur is not None else "",
            }
        )
        stats["kept"] += 1

    return out_rows, dict(stats)


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "split", "audio_path", "text", "duration_s"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "fleurs_eval.yaml"),
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract tar.xz archives, then exit",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    root = Path(cfg["data_root"])
    if not root.is_dir():
        raise SystemExit(f"Missing data root: {root}")

    extract_archives(root, cfg.get("audio", {}).get("archives", []))
    if args.extract_only:
        print("Extract done.")
        return

    index = index_audio_files(root, cfg.get("audio", {}).get("search_roots", ["audio", "."]))
    print(f"Indexed {len(index)} audio path keys under {root}")

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"root": str(root), "splits": {}}
    for split in cfg.get("splits", ["test", "dev"]):
        rows, stats = clean_split(root, split, index, cfg)
        out_tsv = out_dir / f"{split}.tsv"
        write_manifest(out_tsv, rows)
        summary["splits"][split] = {"stats": stats, "manifest": str(out_tsv)}
        print(f"[{split}] kept={stats.get('kept', 0)}  → {out_tsv}")
        for k, v in sorted(stats.items()):
            if k != "kept":
                print(f"         {k}: {v}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
