#!/usr/bin/env python3
"""Clean Kidaw’ida *training* corpus (join cleaned text + audio usability).

Uses frozen cleaned transcripts when present:
  /project/community/rmwisene/tts_data_cleaning/kidawida_cleaned_transcripts.csv

Resolves clips under kidawida_cv27 (Common Voice layout). Does NOT overwrite source.

Outputs:
  /project/community/rmwisene/tts_data_cleaning/kidawida/
    kidawida_cleaned_train.csv
    summary.json
    cleaning_report.txt

  python scripts/clean_kidawida_train.py
  python scripts/clean_kidawida_train.py --limit 200
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

WS_RE = re.compile(r"\s+")


def load_config(path: Path | None) -> dict:
    defaults = {
        "data_root": "/project/community/rmwisene/datasets/kidawida_cv27",
        "cleaned_transcripts": (
            "/project/community/rmwisene/tts_data_cleaning/"
            "kidawida_cleaned_transcripts.csv"
        ),
        "cleaning": {
            "min_chars": 1,
            "max_chars": 2000,
            "min_duration_s": 0.3,
            "max_duration_s": 30.0,
            "drop_missing_audio": True,
            "keep_flagged": True,
        },
        "output": {
            "dir": "/project/community/rmwisene/tts_data_cleaning/kidawida",
        },
    }
    if path is None or not path.is_file():
        return defaults
    if yaml is None:
        raise SystemExit("PyYAML required: pip install pyyaml")
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    out = defaults.copy()
    out.update({k: v for k, v in cfg.items() if k not in ("cleaning", "output")})
    out["cleaning"] = {**defaults["cleaning"], **(cfg.get("cleaning") or {})}
    out["output"] = {**defaults["output"], **(cfg.get("output") or {})}
    return out


def nfc_ws(s: str) -> str:
    t = unicodedata.normalize("NFC", s or "")
    return WS_RE.sub(" ", t).strip()


def is_non_transcript_table(path: Path) -> bool:
    n = path.name.lower()
    if n in {"clip_durations.tsv", "clip_durations.csv", "sizes.txt"}:
        return True
    return any(s in n for s in ("clip_duration", "durations", "duration_ms", "file_size"))


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


def index_clips(root: Path) -> dict[str, Path]:
    """Map clip basename / path → audio file."""
    index: dict[str, Path] = {}
    exts = {".mp3", ".wav", ".flac", ".ogg", ".opus"}
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        # skip if under non-clip junk
        index[p.name] = p
        index[p.stem] = p
        try:
            rel = str(p.relative_to(root))
            index[rel] = p
            index[rel.replace("\\", "/")] = p
        except ValueError:
            pass
    return index


def resolve_clip(path_field: str, index: dict[str, Path], root: Path) -> Path | None:
    if not path_field:
        return None
    raw = path_field.strip()
    candidates = [
        raw,
        Path(raw).name,
        Path(raw).stem,
        f"clips/{Path(raw).name}",
        str(Path("clips") / Path(raw).name),
    ]
    for c in candidates:
        if c in index:
            return index[c]
    direct = root / raw
    if direct.is_file():
        return direct
    clip = root / "clips" / Path(raw).name
    if clip.is_file():
        return clip
    return None


def duration_s(path: Path) -> float | None:
    if sf is None:
        return None
    try:
        info = sf.info(str(path))
        if info.frames and info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        return None
    return None


def load_from_cleaned_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            text = (r.get("cleaned_text") or "").strip()
            original = (r.get("original_text") or text).strip()
            if not text:
                continue
            rows.append(
                {
                    "source_file": r.get("source_file") or "",
                    "path": "",  # filled if we can recover from parallel CV tables
                    "original_text": original,
                    "cleaned_text": nfc_ws(text),
                    "flagged": (r.get("flagged") or "").strip().lower()
                    in {"yes", "true", "1"},
                    "flagged_symbols": r.get("flagged_symbols") or "",
                }
            )
    return rows


def load_cv_tables_with_path(root: Path) -> list[dict]:
    """Load Common Voice TSVs that have both path + sentence."""
    rows: list[dict] = []
    seen: set[str] = set()
    files = sorted(
        {p.resolve() for p in list(root.rglob("*.tsv")) + list(root.rglob("*.csv"))},
        key=lambda p: (
            0 if "train" in p.name.lower() else 1 if "validated" in p.name.lower() else 2,
            str(p),
        ),
    )
    for path in files:
        if is_non_transcript_table(path) or "fleurs" in str(path).lower():
            continue
        sample = path.read_text(encoding="utf-8-sig", errors="replace")[:4096]
        try:
            delim = csv.Sniffer().sniff(sample, delimiters="\t,;|").delimiter
        except Exception:
            delim = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open(encoding="utf-8-sig", newline="", errors="replace") as f:
            reader = csv.DictReader(f, delimiter=delim)
            fields = list(reader.fieldnames or [])
            text_col = detect_col(
                fields, ("sentence", "text", "transcript", "transcription")
            )
            path_col = detect_col(
                fields, ("path", "clip", "filename", "file", "audio", "wav")
            )
            if text_col is None or path_col is None:
                continue
            for r in reader:
                text = (r.get(text_col) or "").strip()
                clip = (r.get(path_col) or "").strip()
                if not text or not clip:
                    continue
                key = f"{clip}::{text}"
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "source_file": str(path.relative_to(root))
                        if path.is_relative_to(root)
                        else str(path),
                        "path": clip,
                        "original_text": text,
                        "cleaned_text": nfc_ws(text.lower()),
                        "flagged": False,
                        "flagged_symbols": "",
                    }
                )
    return rows


def join_cleaned_with_paths(
    cleaned_rows: list[dict], cv_rows: list[dict]
) -> list[dict]:
    """Attach audio path by matching original/cleaned text to CV tables."""
    by_orig: dict[str, list[dict]] = {}
    by_clean: dict[str, list[dict]] = {}
    for r in cv_rows:
        o = nfc_ws(r["original_text"].lower())
        c = nfc_ws(r["cleaned_text"])
        by_orig.setdefault(o, []).append(r)
        by_clean.setdefault(c, []).append(r)

    joined: list[dict] = []
    for r in cleaned_rows:
        o = nfc_ws(r["original_text"].lower())
        c = nfc_ws(r["cleaned_text"])
        cand = None
        if o in by_orig:
            cand = by_orig[o][0]
        elif c in by_clean:
            cand = by_clean[c][0]
        out = dict(r)
        if cand:
            out["path"] = cand["path"]
            out["cv_source"] = cand["source_file"]
        else:
            out["path"] = ""
            out["cv_source"] = ""
        joined.append(out)
    return joined


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "kidawida_train_clean.yaml",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--data_root", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config if args.config.is_file() else None)
    root = args.data_root or Path(cfg["data_root"])
    cleaned_csv = Path(cfg["cleaned_transcripts"])
    out_dir = args.out_dir or Path(cfg["output"]["dir"])
    clean_cfg = cfg["cleaning"]

    if sf is None:
        raise SystemExit("soundfile required: pip install soundfile")
    if not root.is_dir():
        raise SystemExit(f"Missing Kidaw’ida root: {root}")

    print(f"Indexing clips under {root} …")
    index = index_clips(root)
    n_audio = len({p.resolve() for p in index.values()})
    print(f"  audio files: {n_audio}")
    if n_audio:
        sample = sorted({p for p in index.values()}, key=lambda p: p.name)[:5]
        print(f"  sample: {[str(p.relative_to(root)) for p in sample]}")

    print("Loading Common Voice path+sentence tables …")
    cv_rows = load_cv_tables_with_path(root)
    print(f"  CV rows with path: {len(cv_rows)}")

    if cleaned_csv.is_file():
        print(f"Loading frozen cleaned transcripts: {cleaned_csv}")
        cleaned_rows = load_from_cleaned_csv(cleaned_csv)
        print(f"  cleaned text rows: {len(cleaned_rows)}")
        rows = join_cleaned_with_paths(cleaned_rows, cv_rows)
        n_with_path = sum(1 for r in rows if r.get("path"))
        print(f"  joined with audio path: {n_with_path}/{len(rows)}")
    else:
        print("No frozen cleaned CSV — using CV tables directly")
        rows = cv_rows

    if args.limit:
        rows = rows[: args.limit]

    kept: list[dict] = []
    drops: Counter = Counter()
    durations: list[float] = []
    total_dur = 0.0
    n_flagged_kept = 0

    for i, row in enumerate(rows, 1):
        text = row["cleaned_text"]
        flagged = bool(row.get("flagged"))
        if flagged and not clean_cfg.get("keep_flagged", True):
            drops["flagged_excluded"] += 1
            continue

        reason = ""
        audio = resolve_clip(row.get("path") or "", index, root)
        dur = None

        if not text:
            reason = "empty_text"
        elif len(text) < int(clean_cfg["min_chars"]):
            reason = "text_too_short"
        elif len(text) > int(clean_cfg["max_chars"]):
            reason = "text_too_long"
        elif audio is None:
            reason = "missing_audio" if clean_cfg.get("drop_missing_audio", True) else ""
        else:
            dur = duration_s(audio)
            if dur is None:
                reason = "unreadable_audio"
            elif dur < float(clean_cfg["min_duration_s"]):
                reason = "duration_too_short"
            elif dur > float(clean_cfg["max_duration_s"]):
                reason = "duration_too_long"

        if reason:
            drops[reason] += 1
            continue

        assert audio is not None and dur is not None
        durations.append(dur)
        total_dur += dur
        if flagged:
            n_flagged_kept += 1
        kept.append(
            {
                "id": Path(audio).stem,
                "audio_path": str(audio.resolve()),
                "clip": row.get("path") or Path(audio).name,
                "original_text": row["original_text"],
                "cleaned_text": text,
                "duration_s": f"{dur:.3f}",
                "flagged": "yes" if flagged else "no",
                "flagged_symbols": row.get("flagged_symbols") or "",
                "source_file": row.get("source_file") or row.get("cv_source") or "",
            }
        )
        if i % 2000 == 0 or i == len(rows):
            print(f"  processed {i}/{len(rows)}  kept={len(kept)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "kidawida_cleaned_train.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "audio_path",
                "clip",
                "original_text",
                "cleaned_text",
                "duration_s",
                "flagged",
                "flagged_symbols",
                "source_file",
            ],
        )
        w.writeheader()
        w.writerows(kept)

    summary = {
        "data_root": str(root),
        "cleaned_transcripts": str(cleaned_csv) if cleaned_csv.is_file() else None,
        "n_input": len(rows),
        "n_kept": len(kept),
        "n_dropped": len(rows) - len(kept),
        "n_flagged_kept": n_flagged_kept,
        "keep_rate": round(len(kept) / max(len(rows), 1), 6),
        "total_duration_hours": round(total_dur / 3600.0, 4),
        "duration_s_mean": round(sum(durations) / max(len(durations), 1), 4),
        "duration_s_min": round(min(durations), 4) if durations else None,
        "duration_s_max": round(max(durations), 4) if durations else None,
        "drop_reasons": dict(drops),
        "cleaning": clean_cfg,
        "output_csv": str(csv_path),
        "note": (
            "Source kidawida_cv27 not modified. Flagged q/x/ä/~ kept for review. "
            "Compare hours to Kinyarwanda (~6.4 h) when balancing."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report = [
        "Kidaw’ida training corpus — cleaning report",
        "===========================================",
        "",
        f"Source: {root}",
        f"Cleaned text: {cleaned_csv if cleaned_csv.is_file() else '(CV raw)'}",
        f"Output: {out_dir}",
        "Source files were NOT modified.",
        "",
        f"Input rows:     {len(rows)}",
        f"Kept (usable):  {len(kept)}",
        f"Dropped:        {len(rows) - len(kept)}",
        f"Flagged kept:   {n_flagged_kept}",
        f"Keep rate:      {summary['keep_rate']:.2%}",
        f"Total duration: {summary['total_duration_hours']:.3f} h",
        f"Duration mean:  {summary['duration_s_mean']} s",
        "",
        "Drop reasons:",
    ]
    for k, v in sorted(drops.items(), key=lambda kv: -kv[1]):
        report.append(f"  {k}: {v}")
    if not drops:
        report.append("  (none)")
    report += [
        "",
        "Next: balance Kin vs Dav by usable speech hours, then minor frontend fixes.",
        "",
    ]
    (out_dir / "cleaning_report.txt").write_text("\n".join(report), encoding="utf-8")

    print()
    print(f"Input:   {len(rows)}")
    print(f"Kept:    {len(kept)}")
    print(f"Dropped: {len(rows) - len(kept)}")
    print(f"Flagged kept: {n_flagged_kept}")
    print(f"Hours:   {summary['total_duration_hours']:.3f}")
    print(f"Drops:   {dict(drops)}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'cleaning_report.txt'}")


if __name__ == "__main__":
    main()
