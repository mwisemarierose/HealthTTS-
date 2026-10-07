#!/usr/bin/env python3
"""Score TTS synths on FLEURS with agreed metrics:

1) WER & CER (ASR loopback) — NeMo CTC from Drive (.nemo)
2) Latency & RTF (from synthesis.tsv or --retime)
3) UTMOS (naturalness)

  # download CTC first (see scripts/download_ctc_asr.sh)
  python scripts/score_tts_fleurs.py \\
    --synth_dir $BASELINES/eval_fleurs/mms_tts_kin/test \\
    --asr_nemo /project/community/rmwisene/asr/combined-ctc-15-ep-nocl.nemo \\
    --retime --tts_model_dir $BASELINES/mms_tts_kin
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


def load_ctc(nemo_path: Path, device: str):
    import torch
    from nemo.collections.asr.models import EncDecCTCModelBPE

    if not nemo_path.is_file():
        raise FileNotFoundError(
            f"Missing CTC checkpoint: {nemo_path}\n"
            "Download with: bash scripts/download_ctc_asr.sh"
        )
    map_location = "cuda" if device == "cuda" else "cpu"
    model = EncDecCTCModelBPE.restore_from(str(nemo_path), map_location=map_location)
    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    return model


def ctc_transcribe(model, wav_path: Path) -> str:
    out = model.transcribe([str(wav_path)])
    hyp = out[0]
    if hasattr(hyp, "text"):
        return str(hyp.text).strip()
    return str(hyp).strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--synth_dir",
        default=os.environ.get("BASELINES", "/project/community/rmwisene/tts_baselines")
        + "/eval_fleurs/mms_tts_kin/test",
    )
    p.add_argument(
        "--asr_nemo",
        default=os.environ.get(
            "CTC_NEMO",
            "/project/community/rmwisene/asr/combined-ctc-15-ep-nocl.nemo",
        ),
        help="Path to Drive CTC .nemo for ASR loopback",
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

    need_retime = args.retime or all(
        not (r.get("synth_time_s") or "").strip() for r in rows
    )

    tts_model = tts_tok = None
    if need_retime:
        from transformers import AutoTokenizer, VitsModel

        print(f"Loading TTS for timing: {args.tts_model_dir}")
        tts_model = VitsModel.from_pretrained(args.tts_model_dir).to(device)
        tts_tok = AutoTokenizer.from_pretrained(args.tts_model_dir)
        tts_model.eval()

    asr_model = None
    if not args.skip_asr:
        print(f"Loading CTC ASR: {args.asr_nemo}")
        asr_model = load_ctc(Path(args.asr_nemo), device)

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
        if asr_model is not None:
            asr_text = ctc_transcribe(asr_model, hyp_audio)
            refs.append(normalize_for_asr(text))
            hyps.append(normalize_for_asr(asr_text))

        utmos_v = ""
        if utmos is not None:
            wav = torch.tensor(audio, dtype=torch.float32, device=device)
            if sr != 16000:
                import torchaudio

                wav = torchaudio.functional.resample(
                    wav.unsqueeze(0), sr, 16000
                ).squeeze(0)
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
        "asr_nemo": None if args.skip_asr else str(args.asr_nemo),
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
