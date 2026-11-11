#!/usr/bin/env python3
"""Build a blind human-listening set from existing synth folders.

Reads configs/listening_eval.yaml, finds shared utterance ids across
paired synth dirs, samples N items, copies wavs with anonymous names,
writes a listener CSV + answer key (do NOT share the key with listeners).

  python scripts/prepare_listening_set.py
  python scripts/prepare_listening_set.py --pair_id dav_b1_vs_e4 --n 30
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

try:
    import yaml
except ImportError as e:
    raise SystemExit("pip install pyyaml") from e


def load_synth_index(synth_dir: Path) -> dict[str, dict]:
    """id -> row from synthesis.tsv or scores.tsv."""
    scores = synth_dir / "scores.tsv"
    synth = synth_dir / "synthesis.tsv"
    path = scores if scores.is_file() else synth
    if not path.is_file():
        raise SystemExit(f"Missing {scores} or {synth}")
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = {}
    for r in rows:
        uid = (r.get("id") or r.get("utt_id") or "").strip()
        if not uid:
            continue
        hyp = (r.get("hyp_audio") or "").strip()
        if not hyp:
            hyp = str(synth_dir / f"{uid}.wav")
        r = dict(r)
        r["_wav"] = hyp
        r["_utt_id"] = uid
        out[uid] = r
    return out


def utmos(row: dict) -> float | None:
    for k in ("utmos", "UTMOS", "utmos_mean"):
        if k in row and str(row[k]).strip():
            try:
                return float(row[k])
            except ValueError:
                pass
    return None


def duration(row: dict) -> float | None:
    for k in ("duration_s", "duration"):
        if k in row and str(row[k]).strip():
            try:
                return float(row[k])
            except ValueError:
                pass
    return None


def sample_ids(
    idx_a: dict[str, dict],
    idx_b: dict[str, dict],
    n: int,
    seed: int,
    prefer_gap: bool,
    min_d: float,
    max_d: float,
) -> list[str]:
    common = sorted(set(idx_a) & set(idx_b))
    scored: list[tuple[float, str]] = []
    for uid in common:
        wa = Path(idx_a[uid]["_wav"])
        wb = Path(idx_b[uid]["_wav"])
        if not wa.is_file() or not wb.is_file():
            continue
        da, db = duration(idx_a[uid]), duration(idx_b[uid])
        if da is not None and (da < min_d or da > max_d):
            continue
        if db is not None and (db < min_d or db > max_d):
            continue
        ua, ub = utmos(idx_a[uid]), utmos(idx_b[uid])
        gap = abs(ua - ub) if ua is not None and ub is not None else 0.0
        scored.append((gap, uid))
    if prefer_gap:
        scored.sort(key=lambda x: (-x[0], x[1]))
    else:
        rng = random.Random(seed)
        rng.shuffle(scored)
    picked = [uid for _, uid in scored[:n]]
    if len(picked) < n:
        print(f"WARNING: only {len(picked)}/{n} items available")
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "listening_eval.yaml",
    )
    ap.add_argument("--pair_id", default="", help="Only build this pair id (default: all)")
    ap.add_argument("--n", type=int, default=0, help="Override n_items_per_pair")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_root = Path(cfg["output_dir"])
    samp = cfg.get("sampling", {})
    n_default = int(args.n or samp.get("n_items_per_pair", 30))
    seed = int(samp.get("seed", 42))
    prefer_gap = bool(samp.get("prefer_high_utmos_gap", True))
    min_d = float(samp.get("min_duration_s", 1.0))
    max_d = float(samp.get("max_duration_s", 12.0))

    key_rows: list[dict] = []
    listen_rows: list[dict] = []
    item_i = 0

    for lang, pairs in (cfg.get("pairs") or {}).items():
        for pair in pairs:
            pid = pair["id"]
            if args.pair_id and pid != args.pair_id:
                continue
            a = pair["a"]
            b = pair["b"]
            dir_a = Path(a["synth_dir"])
            dir_b = Path(b["synth_dir"])
            print(f"\n=== {lang} / {pid} ===")
            print(f"  A={a['label']}  {dir_a}")
            print(f"  B={b['label']}  {dir_b}")
            if not dir_a.is_dir() or not dir_b.is_dir():
                print("  SKIP — synth dir missing (restore or re-synth first)")
                continue

            idx_a = load_synth_index(dir_a)
            idx_b = load_synth_index(dir_b)
            ids = sample_ids(idx_a, idx_b, n_default, seed, prefer_gap, min_d, max_d)
            pair_dir = out_root / lang / pid / "audio"
            pair_dir.mkdir(parents=True, exist_ok=True)

            rng = random.Random(seed + hash(pid) % 10000)
            for uid in ids:
                item_i += 1
                item_id = f"{pid}_{item_i:04d}"
                # randomize left/right so listeners don't learn system order
                swap = rng.random() < 0.5
                left_lab, right_lab = (b["label"], a["label"]) if swap else (a["label"], b["label"])
                left_src = Path(idx_b[uid]["_wav"] if swap else idx_a[uid]["_wav"])
                right_src = Path(idx_a[uid]["_wav"] if swap else idx_b[uid]["_wav"])
                left_dst = pair_dir / f"{item_id}_L.wav"
                right_dst = pair_dir / f"{item_id}_R.wav"
                shutil.copy2(left_src, left_dst)
                shutil.copy2(right_src, right_dst)

                text = (idx_a[uid].get("text") or idx_b[uid].get("text") or "").strip()
                ua, ub = utmos(idx_a[uid]), utmos(idx_b[uid])
                key_rows.append(
                    {
                        "item_id": item_id,
                        "pair_id": pid,
                        "language": lang,
                        "utt_id": uid,
                        "text": text,
                        "system_A_label": a["label"],
                        "system_B_label": b["label"],
                        "left_is": left_lab,
                        "right_is": right_lab,
                        "utmos_A": ua if ua is not None else "",
                        "utmos_B": ub if ub is not None else "",
                        "wav_L": str(left_dst),
                        "wav_R": str(right_dst),
                    }
                )
                listen_rows.append(
                    {
                        "item_id": item_id,
                        "pair_id": pid,
                        "language": lang,
                        "text": text,
                        "wav_L": str(left_dst),
                        "wav_R": str(right_dst),
                        "naturalness_pref": "",  # L | R | equal
                        "intelligibility_pref": "",
                        "mos_L": "",
                        "mos_R": "",
                        "listener_id": "",
                        "notes": "",
                    }
                )
            print(f"  wrote {len(ids)} items → {pair_dir}")

    out_root.mkdir(parents=True, exist_ok=True)
    key_path = out_root / "answer_key.csv"
    listen_path = out_root / "listener_sheet.csv"
    with key_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()) if key_rows else ["item_id"])
        w.writeheader()
        w.writerows(key_rows)
    with listen_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(listen_rows[0].keys()) if listen_rows else ["item_id"],
        )
        w.writeheader()
        w.writerows(listen_rows)

    meta = {
        "n_items": len(listen_rows),
        "config": str(args.config),
        "answer_key": str(key_path),
        "listener_sheet": str(listen_path),
        "note": "Do NOT give answer_key.csv to listeners. Collect prefs in listener_sheet or a form.",
    }
    (out_root / "listening_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("\nDone.")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
