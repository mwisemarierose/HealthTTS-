#!/usr/bin/env python3
"""Score TTS synths on FLEURS with agreed metrics:

1) WER & CER (ASR loopback)
2) Latency & RTF (from synthesis.tsv or --retime)
3) UTMOS (naturalness)

  python scripts/score_tts_fleurs.py \\
    --synth_dir $BASELINES/eval_fleurs/mms_tts_kin/test \\
    --asr_model mbazaNLP/Whisper-Small-Kinyarwanda

  # if synthesis.tsv has no timing columns, measure RTF by re-synthesizing:
  python scripts/score_tts_fleurs.py ... --retime --tts_model_dir $BASELINES/mms_tts_kin
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import unicodedata
from pathlib import Path


def normalize_for_asr(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "")
    t = t.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def read_tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--synth_dir",
        default=os.environ.get("BASELINES", "/project/community/rmwisene/tts_baselines")
        + "/eval_fleurs/mms_tts_kin/test",
    )
    p.add_argument(
        "--asr_model",
        default="mbazaNLP/Whisper-Small-Kinyarwanda",
        help="HF ASR model for loopback WER/CER",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--retime",
        action="store_true",
        help="Re-run MMS TTS timed to fill latency/RTF",
    )
    p.add_argument(
        "--tts_model_dir",
        default=os.environ.get("BASELINES", "/project/community/rmwisene/tts_baselines")
        + "/mms_tts_kin",
    )
    p.add_argument("--skip_utmos", action="store_true")
    p.add_argument("--skip_asr", action="store_true")
    args = p.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    from jiwer import cer as jiwer_cer
    from jiwer import wer as jiwer_wer

    synth_dir = Path(args.synth_dir)
    meta_path = synth_dir / "synthesis.tsv"
    if not meta_path.is_file():
        raise SystemExit(f"Missing {meta_path}")

    rows = read_tsv(meta_path)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — using CPU")
        device = "cpu"

    # --- optional retime (latency / RTF) ---
    tts_model = tts_tok = None
    if args.retime or any(not r.get("synth_time_s") for r in rows):
        need_retime = args.retime or all(not (r.get("synth_time_s") or "").strip() for r in rows)
    else:
        need_retime = False

    if need_retime:
        from transformers import AutoTokenizer, VitsModel

        print(f"Loading TTS for timing: {args.tts_model_dir}")
        tts_model = VitsModel.from_pretrained(args.tts_model_dir).to(device)
        tts_tok = AutoTokenizer.from_pretrained(args.tts_model_dir)
        tts_model.eval()

    # --- ASR ---
    asr_pipe = None
    if not args.skip_asr:
        from transformers import pipeline

        print(f"Loading ASR: {args.asr_model}")
        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=args.asr_model,
            device=0 if device == "cuda" else -1,
        )

    # --- UTMOS ---
    utmos = None
    if not args.skip_utmos:
        print("Loading UTMOS22 Strong …")
        utmos = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True,
        )
        utmos = utmos.to(device)
        utmos.eval()

    refs, hyps = [], []
    score_rows = []
    latencies, rtfs, utmos_scores = [], [], []

    for i, row in enumerate(rows, 1):
        utt_id = row["id"]
        text = row["text"]
        hyp_audio = Path(row["hyp_audio"])
        if not hyp_audio.is_file():
            hyp_audio = synth_dir / f"{utt_id}.wav"
        if not hyp_audio.is_file():
            print(f"  skip missing audio: {utt_id}")
            continue

        audio, sr = sf.read(str(hyp_audio))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        duration_s = float(len(audio) / sr)

        synth_time = row.get("synth_time_s", "").strip()
        rtf = row.get("rtf", "").strip()

        if need_retime and tts_model is not None:
            inputs = tts_tok(text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = tts_model(**inputs).waveform
            if device == "cuda":
                torch.cuda.synchronize()
            synth_time_f = time.perf_counter() - t0
            # use existing audio duration for RTF (same text)
            rtf_f = synth_time_f / duration_s if duration_s > 0 else float("nan")
            synth_time = f"{synth_time_f:.4f}"
            rtf = f"{rtf_f:.4f}"
            latencies.append(synth_time_f)
            rtfs.append(rtf_f)
        elif synth_time:
            try:
                st = float(synth_time)
                latencies.append(st)
                if rtf:
                    rtfs.append(float(rtf))
                elif duration_s > 0:
                    rtfs.append(st / duration_s)
            except ValueError:
                pass

        asr_text = ""
        if asr_pipe is not None:
            out = asr_pipe(str(hyp_audio))
            asr_text = out["text"] if isinstance(out, dict) else str(out)
            refs.append(normalize_for_asr(text))
            hyps.append(normalize_for_asr(asr_text))

        utmos_v = ""
        if utmos is not None:
            wav = torch.tensor(audio, dtype=torch.float32, device=device)
            if sr != 16000:
                import torchaudio

                wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
            with torch.no_grad():
                score = utmos(wav.unsqueeze(0), 16000)
            utmos_f = float(score.item() if hasattr(score, "item") else score)
            utmos_v = f"{utmos_f:.4f}"
            utmos_scores.append(utmos_f)

        score_rows.append(
            {
                "id": utt_id,
                "text": text,
                "asr_text": asr_text,
                "hyp_audio": str(hyp_audio),
                "duration_s": f"{duration_s:.3f}",
                "synth_time_s": synth_time,
                "rtf": rtf,
                "utmos": utmos_v,
            }
        )
        if i % 20 == 0 or i == len(rows):
            print(f"  scored [{i}/{len(rows)}] {utt_id}")

    metrics = {
        "n": len(score_rows),
        "synth_dir": str(synth_dir),
        "asr_model": None if args.skip_asr else args.asr_model,
        "wer": None,
        "cer": None,
        "latency_s_mean": float(np.mean(latencies)) if latencies else None,
        "latency_s_median": float(np.median(latencies)) if latencies else None,
        "rtf_mean": float(np.mean(rtfs)) if rtfs else None,
        "rtf_median": float(np.median(rtfs)) if rtfs else None,
        "utmos_mean": float(np.mean(utmos_scores)) if utmos_scores else None,
        "utmos_median": float(np.median(utmos_scores)) if utmos_scores else None,
    }
    if refs and hyps:
        metrics["wer"] = float(jiwer_wer(refs, hyps))
        metrics["cer"] = float(jiwer_cer(refs, hyps))

    scores_path = synth_dir / "scores.tsv"
    with scores_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "id",
            "text",
            "asr_text",
            "hyp_audio",
            "duration_s",
            "synth_time_s",
            "rtf",
            "utmos",
        ]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(score_rows)

    metrics_path = synth_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print("\n=== Metrics ===")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {scores_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
