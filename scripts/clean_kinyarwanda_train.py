#!/usr/bin/env python3
"""Clean Kinyarwanda TTS *training* corpus (text + audio usability).

Source (never overwritten):
  /project/community/rmwisene/datasets/kinyarwanda_tts/
    tts-dataset.csv   # lines: TTS_1_2 "transcript..."
    audio/

Outputs:
  /project/community/rmwisene/tts_data_cleaning/kinyarwanda/
    kinyarwanda_cleaned_train.csv
    cleaning_report.txt
    summary.json

  python scripts/clean_kinyarwanda_train.py
  python scripts/clean_kinyarwanda_train.py --config configs/kinyarwanda_train_clean.yaml
  python scripts/clean_kinyarwanda_train.py --limit 100   # smoke
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

# Metadata line: TTS_1_2 "transcript..."  or  TTS_1_2,transcript
LINE_QUOTED = re.compile(r'^\s*(\S+)\s+"(.*)"\s*$')
LINE_CSV = re.compile(r'^\s*([^,]+),\s*"(.*)"\s*$|^\s*([^,]+),\s*(.+)\s*$')
WS_RE = re.compile(r"\s+")


def load_config(path: Path | None) -> dict:
    defaults = {
        "data_root": "/project/community/rmwisene/datasets/kinyarwanda_tts",
        "metadata": "tts-dataset.csv",
        "audio_dir": "audio",
        "cleaning": {
            "min_chars": 1,
            "max_chars": 2000,
            "min_duration_s": 0.3,
            "max_duration_s": 30.0,
            "drop_missing_audio": True,
            "normalize_unicode_nfc": True,
            "collapse_whitespace": True,
            "lowercase": True,
        },
        "output": {
            "dir": "/project/community/rmwisene/tts_data_cleaning/kinyarwanda",
        },
    }
    if path is None or not path.is_file():
        return defaults
    if yaml is None:
        raise SystemExit("PyYAML required: pip install pyyaml")
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # shallow merge
    out = defaults.copy()
    out.update({k: v for k, v in cfg.items() if k not in ("cleaning", "output")})
    out["cleaning"] = {**defaults["cleaning"], **(cfg.get("cleaning") or {})}
    out["output"] = {**defaults["output"], **(cfg.get("output") or {})}
    return out


def clean_text(text: str, cfg: dict) -> str:
    t = text or ""
    if cfg.get("normalize_unicode_nfc", True):
        t = unicodedata.normalize("NFC", t)
    t = "".join(QUOTE_MAP.get(ch, ch) for ch in t)
    if cfg.get("lowercase", True):
        t = t.lower()
    if cfg.get("collapse_whitespace", True):
        t = WS_RE.sub(" ", t).strip()
    else:
        t = t.strip()
    return t


def parse_metadata(path: Path) -> list[dict]:
    """Parse tts-dataset.csv → [{id, original_text}, ...]."""
    rows: list[dict] = []
    with path.open(encoding="utf-8-sig", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("license") or low.startswith("http"):
                continue

            utt_id = None
            text = None
            m = LINE_QUOTED.match(line)
            if m:
                utt_id, text = m.group(1), m.group(2)
            else:
                m = LINE_CSV.match(line)
                if m:
                    utt_id = m.group(1) or m.group(3)
                    text = m.group(2) if m.group(2) is not None else m.group(4)

            if not utt_id or text is None:
                # last resort: first token + rest
                parts = line.split(None, 1)
                if len(parts) == 2:
                    utt_id, text = parts[0], parts[1].strip().strip('"')
                else:
                    continue

            text = text.strip()
            if not text or re.fullmatch(r"[0-9]+([.,][0-9]+)?", text):
                continue
            rows.append({"id": utt_id.strip(), "original_text": text, "line_no": line_no})
    return rows


def index_audio(audio_dir: Path) -> dict[str, Path]:
    """Map id / stem / filename → path."""
    index: dict[str, Path] = {}
    if not audio_dir.is_dir():
        return index
    exts = {".wav", ".flac", ".mp3", ".ogg", ".opus"}
    for p in audio_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        index[p.name] = p
        index[p.stem] = p
        # common: TTS_1_2.wav ↔ id TTS_1_2
        index[p.stem.lower()] = p
    return index


def _norm_id(s: str) -> str:
    """Normalize ids for fuzzy audio matching."""
    t = (s or "").strip().lower()
    t = t.replace("\\", "/")
    for ext in (".wav", ".flac", ".mp3", ".ogg", ".opus"):
        if t.endswith(ext):
            t = t[: -len(ext)]
            break
    # mbazaNLP: metadata TTS_1_2 ↔ audio "TTS 1_2.wav"
    t = re.sub(r"^tts[\s_]+", "tts_", t)
    t = re.sub(r"[\s\-]+", "_", t)
    return t


def meta_id_to_audio_stem(utt_id: str) -> list[str]:
    """Map metadata id TTS_1_2 → audio stems like 'TTS 1_2'."""
    raw = (utt_id or "").strip()
    stems = [raw, Path(raw).stem]
    # TTS_1_2 → TTS 1_2  (space after TTS)
    if re.match(r"^TTS_\d", raw, flags=re.I):
        stems.append(re.sub(r"^TTS_", "TTS ", raw, count=1, flags=re.I))
    # TTS 1_2 → TTS_1_2
    if re.match(r"^TTS\s+\d", raw, flags=re.I):
        stems.append(re.sub(r"^TTS\s+", "TTS_", raw, count=1, flags=re.I))
    return list(dict.fromkeys(stems))  # unique, preserve order


def resolve_audio(utt_id: str, index: dict[str, Path], audio_dir: Path) -> Path | None:
    stems = meta_id_to_audio_stem(utt_id)
    candidates: list[str] = []
    for stem in stems:
        candidates.extend(
            [
                stem,
                stem.lower(),
                f"{stem}.wav",
                f"{stem}.flac",
                f"{stem}.mp3",
                Path(stem).name,
            ]
        )
    for c in candidates:
        if c in index:
            return index[c]
    for stem in stems:
        for ext in (".wav", ".flac", ".mp3"):
            direct = audio_dir / f"{stem}{ext}"
            if direct.is_file():
                return direct
    return None


def audio_duration_s(path: Path) -> float | None:
    if sf is None:
        return None
    try:
        info = sf.info(str(path))
        if info.frames and info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean Kinyarwanda TTS training corpus")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "kinyarwanda_train_clean.yaml",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--data_root", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config if args.config.is_file() else None)
    root = args.data_root or Path(cfg["data_root"])
    meta = root / cfg["metadata"]
    audio_dir = root / cfg["audio_dir"]
    out_dir = args.out_dir or Path(cfg["output"]["dir"])
    clean_cfg = cfg["cleaning"]

    if not meta.is_file():
        raise SystemExit(f"Missing metadata: {meta}")
    if sf is None:
        raise SystemExit("soundfile required: pip install soundfile")

    print(f"Parsing {meta} …")
    rows = parse_metadata(meta)
    if args.limit:
        rows = rows[: args.limit]
    print(f"  metadata rows: {len(rows)}")

    print(f"Indexing audio under {audio_dir} …")
    index = index_audio(audio_dir)
    unique_audio = sorted({p.resolve() for p in index.values()}, key=lambda p: p.name)
    print(f"  audio files: {len(unique_audio)}")
    if rows:
        print(f"  sample meta ids: {[r['id'] for r in rows[:5]]}")
    if unique_audio:
        print(f"  sample audio:    {[p.name for p in unique_audio[:5]]}")

    # Build flexible matchers (mbazaNLP often uses nested dirs / different stems)
    by_stem: dict[str, Path] = {}
    by_stem_norm: dict[str, Path] = {}
    for p in unique_audio:
        by_stem[p.stem] = p
        by_stem_norm[_norm_id(p.stem)] = p
        # also index parent/stem style keys
        try:
            rel = p.relative_to(audio_dir)
            by_stem[str(rel.with_suffix(""))] = p
            by_stem_norm[_norm_id(str(rel.with_suffix("")))] = p
        except ValueError:
            pass

    kept_rows: list[dict] = []
    drop_reasons: Counter = Counter()
    durations: list[float] = []
    total_dur = 0.0
    match_debug_left = 5

    for i, row in enumerate(rows, 1):
        utt_id = row["id"]
        original = row["original_text"]
        cleaned = clean_text(original, clean_cfg)

        reason = ""
        audio = resolve_audio(utt_id, index, audio_dir)
        if audio is None:
            # try normalized / relative keys
            nid = _norm_id(utt_id)
            audio = by_stem.get(utt_id) or by_stem_norm.get(nid)
            if audio is None:
                # id may be like audio/TTS_1_2 or path without extension
                stem = Path(utt_id).stem
                audio = by_stem.get(stem) or by_stem_norm.get(_norm_id(stem))
        if audio is None and match_debug_left > 0:
            print(f"  [debug] no audio for id={utt_id!r} norm={_norm_id(utt_id)!r}")
            match_debug_left -= 1
        dur = None

        if not cleaned:
            reason = "empty_text"
        elif len(cleaned) < int(clean_cfg["min_chars"]):
            reason = "text_too_short"
        elif len(cleaned) > int(clean_cfg["max_chars"]):
            reason = "text_too_long"
        elif audio is None:
            if clean_cfg.get("drop_missing_audio", True):
                reason = "missing_audio"
            else:
                reason = ""
        else:
            dur = audio_duration_s(audio)
            if dur is None:
                reason = "unreadable_audio"
            elif dur < float(clean_cfg["min_duration_s"]):
                reason = "duration_too_short"
            elif dur > float(clean_cfg["max_duration_s"]):
                reason = "duration_too_long"

        if reason:
            drop_reasons[reason] += 1
            continue

        assert audio is not None and dur is not None
        durations.append(dur)
        total_dur += dur
        kept_rows.append(
            {
                "id": utt_id,
                "audio_path": str(audio.resolve()),
                "original_text": original,
                "cleaned_text": cleaned,
                "duration_s": f"{dur:.3f}",
                "changed": "yes" if cleaned != original else "no",
            }
        )
        if i % 500 == 0 or i == len(rows):
            print(f"  processed {i}/{len(rows)}  kept={len(kept_rows)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "kinyarwanda_cleaned_train.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "audio_path",
                "original_text",
                "cleaned_text",
                "duration_s",
                "changed",
            ],
        )
        w.writeheader()
        w.writerows(kept_rows)

    n_in = len(rows)
    n_kept = len(kept_rows)
    n_dropped = n_in - n_kept
    summary = {
        "data_root": str(root),
        "metadata": str(meta),
        "audio_dir": str(audio_dir),
        "n_input": n_in,
        "n_kept": n_kept,
        "n_dropped": n_dropped,
        "keep_rate": round(n_kept / max(n_in, 1), 6),
        "total_duration_hours": round(total_dur / 3600.0, 4),
        "duration_s_mean": round(sum(durations) / max(len(durations), 1), 4),
        "duration_s_min": round(min(durations), 4) if durations else None,
        "duration_s_max": round(max(durations), 4) if durations else None,
        "drop_reasons": dict(drop_reasons),
        "cleaning": clean_cfg,
        "output_csv": str(csv_path),
        "note": "FLEURS not used. Source kinyarwanda_tts files were not modified.",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report = [
        "Kinyarwanda TTS training corpus — cleaning report",
        "=================================================",
        "",
        f"Source:  {root}",
        f"Metadata:{meta}",
        f"Audio:   {audio_dir}",
        f"Output:  {out_dir}",
        "Source files were NOT modified.",
        "",
        f"Input utterances:  {n_in}",
        f"Kept (usable):     {n_kept}",
        f"Dropped:           {n_dropped}",
        f"Keep rate:         {summary['keep_rate']:.2%}",
        f"Total duration:    {summary['total_duration_hours']:.3f} h",
        f"Duration mean:     {summary['duration_s_mean']} s",
        f"Duration range:    {summary['duration_s_min']} – {summary['duration_s_max']} s",
        "",
        "Drop reasons:",
    ]
    for k, v in sorted(drop_reasons.items(), key=lambda kv: -kv[1]):
        report.append(f"  {k}: {v}")
    if not drop_reasons:
        report.append("  (none)")
    report += [
        "",
        "Text rules: NFC, Unicode quote/dash map, lowercase, whitespace collapse.",
        "Audio rules: must resolve under audio/; duration in "
        f"[{clean_cfg['min_duration_s']}, {clean_cfg['max_duration_s']}] s.",
        "",
        "Next: clean Kidaw’ida training audio+text the same way, then balance by duration.",
        "",
    ]
    (out_dir / "cleaning_report.txt").write_text("\n".join(report), encoding="utf-8")

    print()
    print(f"Input:   {n_in}")
    print(f"Kept:    {n_kept}")
    print(f"Dropped: {n_dropped}")
    print(f"Hours:   {summary['total_duration_hours']:.3f}")
    print(f"Drops:   {dict(drop_reasons)}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'cleaning_report.txt'}")


if __name__ == "__main__":
    main()
