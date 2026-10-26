#!/usr/bin/env python3
"""Prepare merged Kinyarwanda TTS corpus: mbaza + AfriSpeak Female + Male.

Order:
  1) Load mbaza  2) Load AfriSpeak F/M  3) Mild text + 16 kHz export
  4) Merge  5) QC (identical-audio only)  6) stratified train/dev/test
  7) Same manifests for Kinya-Flex and MMS-TTS-Kin

Does NOT modify raw datasets. Audio copies under output/audio/.

  python scripts/prepare_kinyarwanda_tts_corpus.py
  python scripts/prepare_kinyarwanda_tts_corpus.py --limit 50
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
LINE_QUOTED = re.compile(r'^\s*(\S+)\s+"(.*)"\s*$')
SCHEMA = ("utt_id", "audio_path", "text", "speaker_id", "source", "duration")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def mild_text(text: str, cfg: dict) -> str:
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


def count_text_duplicates(rows: list[dict], clean_cfg: dict) -> dict:
    counts: Counter[str] = Counter()
    for r in rows:
        counts[mild_text(r.get("text") or "", clean_cfg)] += 1
    return {
        "n_unique_texts": len(counts),
        "n_texts_with_repeats": sum(1 for c in counts.values() if c > 1),
        "n_extra_rows_same_text": sum(c - 1 for c in counts.values() if c > 1),
    }


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
        for group in by_src.values():
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


def index_audio_dir(audio_dir: Path) -> dict[str, Path]:
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


def parse_afrispeak_manifests(root: Path, manifest_names: list[str]) -> list[dict]:
    """Load AfriSpeak TSV rows: text + audio_name → audio/{id}.wav."""
    rows: list[dict] = []
    seen: set[str] = set()
    for name in manifest_names:
        path = root / name
        if not path.is_file():
            print(f"  warn: missing manifest {path}")
            continue
        with path.open(encoding="utf-8-sig", newline="", errors="replace") as f:
            sample = f.read(4096)
            f.seek(0)
            delim = "\t" if sample.count("\t") >= sample.count(",") else ","
            reader = csv.DictReader(f, delimiter=delim)
            fields = list(reader.fieldnames or [])
            text_col = detect_col(fields, ("text", "sentence", "transcript"))
            audio_col = detect_col(fields, ("audio_name", "audio", "path", "filename", "wav"))
            if text_col is None or audio_col is None:
                raise SystemExit(f"AfriSpeak manifest missing text/audio columns: {path} {fields}")
            for r in reader:
                text = (r.get(text_col) or "").strip()
                audio_name = (r.get(audio_col) or "").strip()
                if not text or not audio_name:
                    continue
                # normalize id (manifest may be int-like)
                audio_id = str(audio_name).strip()
                if audio_id.endswith(".wav"):
                    audio_id = Path(audio_id).stem
                key = f"{name}::{audio_id}"
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "id": audio_id,
                        "original_text": text,
                        "manifest": name,
                    }
                )
    return rows


def resolve_afrispeak_audio(audio_id: str, index: dict[str, Path], audio_dir: Path) -> Path | None:
    for key in (audio_id, f"{audio_id}.wav", str(int(audio_id)) if audio_id.isdigit() else None):
        if key and key in index:
            return index[key]
    for ext in (".wav", ".flac", ".mp3"):
        p = audio_dir / f"{audio_id}{ext}"
        if p.is_file():
            return p
        if audio_id.isdigit():
            p2 = audio_dir / f"{int(audio_id)}{ext}"
            if p2.is_file():
                return p2
    return None


def export_rows(
    items: list[dict],
    *,
    source: str,
    speaker_id: str,
    resolve_fn,
    audio_subdir: Path,
    clean_cfg: dict,
    sr_out: int,
    rejected: list[dict],
    skip_audio_export: bool,
    limit: int,
) -> list[dict]:
    kept: list[dict] = []
    for i, item in enumerate(items):
        if limit and len(kept) >= limit:
            break
        raw_id = str(item["id"])
        safe_id = re.sub(r"[^\w.\-]+", "_", raw_id)
        utt_id = f"{source}_{safe_id}"
        text = mild_text(item["original_text"], clean_cfg)

        if clean_cfg.get("drop_empty_text", True) and not text:
            reject([{"utt_id": utt_id, "source": source, "text": "", "audio_path": ""}], "empty_text", rejected)
            continue
        if len(text) < int(clean_cfg["min_chars"]) or len(text) > int(clean_cfg["max_chars"]):
            reject(
                [{"utt_id": utt_id, "source": source, "text": text, "audio_path": ""}],
                "text_length",
                rejected,
            )
            continue

        src_audio = resolve_fn(raw_id)
        if src_audio is None:
            reject(
                [{"utt_id": utt_id, "source": source, "text": text, "audio_path": ""}],
                "missing_audio",
                rejected,
            )
            continue

        arr, sr_in = read_audio(src_audio)
        if arr is None or not sr_in:
            reject(
                [{"utt_id": utt_id, "source": source, "text": text, "audio_path": str(src_audio)}],
                "unreadable_audio",
                rejected,
            )
            continue

        if skip_audio_export and int(sr_in) == sr_out and src_audio.suffix.lower() == ".wav":
            out_path = src_audio
            dur = float(len(arr) / sr_in)
        else:
            arr = resample_mono(arr, int(sr_in), sr_out)
            out_path = audio_subdir / f"{safe_id}.wav"
            dur = write_wav(out_path, arr, sr_out)

        if dur < float(clean_cfg["min_duration_s"]) or dur > float(clean_cfg["max_duration_s"]):
            reject(
                [{"utt_id": utt_id, "source": source, "text": text, "audio_path": str(out_path)}],
                "duration_out_of_range",
                rejected,
            )
            continue

        kept.append(
            {
                "utt_id": utt_id,
                "audio_path": str(Path(out_path).resolve()),
                "text": text,
                "speaker_id": speaker_id,
                "source": source,
                "duration": round(float(dur), 3),
            }
        )
        if (i + 1) % 500 == 0:
            print(f"    {source} processed {i + 1}/{len(items)}  kept={len(kept)}")
    return kept


def write_review_sample(path: Path, rows: list[dict], n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    idx = idx[: min(n, len(idx))]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["utt_id", "text", "duration", "speaker_id", "source", "audio_path"]
        )
        w.writeheader()
        for j in idx:
            r = rows[j]
            w.writerow(
                {
                    "utt_id": r["utt_id"],
                    "text": r["text"],
                    "duration": r["duration"],
                    "speaker_id": r["speaker_id"],
                    "source": r["source"],
                    "audio_path": r["audio_path"],
                }
            )


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
        help="Keep original wav paths when already 16 kHz",
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
    pool_parts: list[dict] = []

    # ------------------------------------------------------------------
    # 1) mbaza
    # ------------------------------------------------------------------
    print("=== 1/3 mbaza TTS ===")
    mbaza_cfg = cfg["mbaza"]
    root = Path(mbaza_cfg["data_root"])
    meta = root / mbaza_cfg["metadata"]
    audio_dir = root / mbaza_cfg["audio_dir"]
    if not meta.is_file():
        raise SystemExit(f"Missing mbaza metadata: {meta}")
    mbaza_meta = parse_mbaza_metadata(meta)
    index = index_audio_dir(audio_dir)
    print(f"  metadata: {len(mbaza_meta)}  audio indexed: {len(set(index.values()))}")
    mbaza_rows = export_rows(
        mbaza_meta,
        source=mbaza_cfg["source"],
        speaker_id=mbaza_cfg["speaker_id"],
        resolve_fn=lambda uid: resolve_mbaza_audio(uid, index, audio_dir),
        audio_subdir=audio_out / "mbaza",
        clean_cfg=clean_cfg,
        sr_out=sr_out,
        rejected=rejected,
        skip_audio_export=args.skip_audio_export,
        limit=args.limit,
    )
    print(f"  mbaza usable: {len(mbaza_rows)}")
    pool_parts.extend(mbaza_rows)

    # ------------------------------------------------------------------
    # 2–3) AfriSpeak Female / Male
    # ------------------------------------------------------------------
    for key, label in (
        ("afrispeak_female", "2/3 AfriSpeak Female"),
        ("afrispeak_male", "3/3 AfriSpeak Male"),
    ):
        print(f"=== {label} ===")
        acfg = cfg[key]
        aroot = Path(acfg["data_root"])
        if not aroot.is_dir():
            raise SystemExit(
                f"Missing {key} at {aroot}\n"
                f"  hf download {acfg['hf_id']} --repo-type dataset --local-dir {aroot}"
            )
        aaudio = aroot / acfg.get("audio_dir", "audio")
        aindex = index_audio_dir(aaudio)
        items = parse_afrispeak_manifests(aroot, list(acfg.get("manifests") or []))
        print(f"  manifests rows: {len(items)}  audio indexed: {len(set(aindex.values()))}")
        rows = export_rows(
            items,
            source=acfg["source"],
            speaker_id=acfg["speaker_id"],
            resolve_fn=lambda uid, idx=aindex, ad=aaudio: resolve_afrispeak_audio(uid, idx, ad),
            audio_subdir=audio_out / acfg["source"],
            clean_cfg=clean_cfg,
            sr_out=sr_out,
            rejected=rejected,
            skip_audio_export=args.skip_audio_export,
            limit=args.limit,
        )
        print(f"  {key} usable: {len(rows)}")
        pool_parts.extend(rows)

    # ------------------------------------------------------------------
    # Merge + QC
    # ------------------------------------------------------------------
    print("=== Merge + QC ===")
    pool = list(pool_parts)
    print(f"  pool before audio dedupe: {len(pool)}")
    text_dup_stats = count_text_duplicates(pool, clean_cfg)
    print(
        "  text repeats (kept): "
        f"{text_dup_stats['n_texts_with_repeats']} texts × "
        f"+{text_dup_stats['n_extra_rows_same_text']} rows"
    )

    if clean_cfg.get("drop_duplicate_text", False):
        seen: set[str] = set()
        deduped = []
        for r in pool:
            key = mild_text(r["text"], clean_cfg)
            if key in seen:
                reject([r], "duplicate_text", rejected)
                continue
            seen.add(key)
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

    # Review samples
    review_n = int(cfg.get("review", {}).get("n_samples_per_source", 50))
    review_seed = int(cfg.get("review", {}).get("seed", 42))
    by_src: dict[str, list[dict]] = {}
    for r in pool:
        by_src.setdefault(r["source"], []).append(r)
    for src, rows in by_src.items():
        write_review_sample(
            out_dir / f"review_{src}.csv", rows, review_n, review_seed
        )
        print(f"  review sample {src}: {min(review_n, len(rows))} → review_{src}.csv")

    # Split
    print("=== Split ===")
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
        hours = sum(float(r["duration"]) for r in rows) / 3600.0
        print(f"  {name}: {len(rows)} clips, {hours:.3f} h")
    write_manifest(man_dir / "all.csv", pool)

    rej_path = out_dir / "rejected.csv"
    with rej_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["utt_id", "source", "reason", "text", "audio_path", "audio_md5"]
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
        "by_source": src_stats(pool),
        "text_duplicate_report": text_dup_stats,
        "splits": {
            k: {
                "n": len(v),
                "hours": round(sum(float(r["duration"]) for r in v) / 3600.0, 4),
                "by_source": src_stats(v),
            }
            for k, v in splits.items()
        },
        "n_rejected": len(rejected),
        "reject_reasons": dict(Counter(r["reason"] for r in rejected)),
        "audio": cfg["audio"],
        "cleaning": clean_cfg,
        "note": (
            "mbaza + AfriSpeak Female + AfriSpeak Male. No Pastor. "
            "Duplicate text kept; only identical audio bytes dropped. "
            "Use manifests for both Flex and MMS."
        ),
    }
    (out_dir / "dataset_statistics.json").write_text(
        json.dumps(all_stats, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "split_config.json").write_text(
        json.dumps(
            {
                "seed": sp["seed"],
                "train_ratio": sp["train_ratio"],
                "dev_ratio": sp["dev_ratio"],
                "test_ratio": sp["test_ratio"],
                "stratify_by_source": sp.get("stratify_by_source", True),
                "sample_rate": sr_out,
                "output_dir": str(out_dir),
                "manifests": {
                    "train": str(man_dir / "train.csv"),
                    "dev": str(man_dir / "dev.csv"),
                    "test": str(man_dir / "test.csv"),
                    "all": str(man_dir / "all.csv"),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report = [
        "Kinyarwanda TTS merged corpus (mbaza + AfriSpeak)",
        "=================================================",
        f"Output: {out_dir}",
        f"Merged pool: {len(pool)} clips / {all_stats['pool_hours']} h",
        f"Rejected: {len(rejected)}",
        "",
        "By source:",
    ]
    for src, st in all_stats["by_source"].items():
        report.append(f"  {src}: {st['n']} clips / {st['duration_hours']} h")
    report += ["", "Splits:"]
    for k, v in splits.items():
        report.append(f"  {k}: {len(v)} / {all_stats['splits'][k]['hours']} h")
    report += [
        "",
        "Use the same manifests for Kinya-Flex and MMS-TTS-Kin.",
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
