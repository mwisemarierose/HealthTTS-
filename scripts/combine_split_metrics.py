#!/usr/bin/env python3
"""Combine test + dev scores into one metrics.json per baseline.

  python scripts/combine_split_metrics.py \\
    --eval_root $BASELINES/eval_fleurs/mms_tts_kin

Writes:
  <eval_root>/all/scores.tsv
  <eval_root>/all/metrics.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer


def read_tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--eval_root",
        required=True,
        help="e.g. .../eval_fleurs/mms_tts_kin (must contain test/ and dev/)",
    )
    p.add_argument(
        "--splits",
        default="test,dev",
        help="Comma-separated split dirs to merge (default: test,dev)",
    )
    p.add_argument("--out_name", default="all", help="Output subdir name (default: all)")
    args = p.parse_args()

    root = Path(args.eval_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    rows: list[dict] = []
    sources: list[str] = []
    for sp in splits:
        scores = root / sp / "scores.tsv"
        metrics = root / sp / "metrics.json"
        if not scores.is_file():
            raise FileNotFoundError(f"Missing {scores}")
        part = read_tsv(scores)
        for r in part:
            r = dict(r)
            r["split"] = sp
            rows.append(r)
        sources.append(str(metrics if metrics.is_file() else scores))

    refs = [r.get("text", "") for r in rows]
    hyps = [r.get("asr_text", "") for r in rows]

    def floats(key: str) -> list[float]:
        out: list[float] = []
        for r in rows:
            v = (r.get(key) or "").strip()
            if not v:
                continue
            try:
                out.append(float(v))
            except ValueError:
                pass
        return out

    latencies = floats("synth_time_s")
    rtfs = floats("rtf")
    utmos_scores = floats("utmos")

    metrics = {
        "n": len(rows),
        "splits": splits,
        "eval_root": str(root),
        "sources": sources,
        "wer": float(jiwer_wer(refs, hyps)) if refs and any(hyps) else None,
        "cer": float(jiwer_cer(refs, hyps)) if refs and any(hyps) else None,
        "latency_s_mean": float(np.mean(latencies)) if latencies else None,
        "latency_s_median": float(np.median(latencies)) if latencies else None,
        "rtf_mean": float(np.mean(rtfs)) if rtfs else None,
        "rtf_median": float(np.median(rtfs)) if rtfs else None,
        "utmos_mean": float(np.mean(utmos_scores)) if utmos_scores else None,
        "utmos_median": float(np.median(utmos_scores)) if utmos_scores else None,
    }

    out_dir = root / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "scores.tsv"
    fields = [
        "split",
        "id",
        "text",
        "asr_text",
        "hyp_audio",
        "duration_s",
        "synth_time_s",
        "rtf",
        "utmos",
    ]
    with scores_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {scores_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
