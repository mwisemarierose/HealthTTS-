#!/usr/bin/env python3
"""Compare Kin/Dav *training* corpora and test MMS + Flex frontends on Dav text.

FLEURS is evaluation-only and is NOT used here.

By default Kidaw’ida text comes from the FROZEN cleaned corpus:
  /project/community/rmwisene/tts_data_cleaning/kidawida_cleaned_transcripts.csv
(use --use_raw_dav to read kidawida_cv27 instead).

  python scripts/freeze_cleaned_kidawida.py   # once
  python scripts/analyze_frontend_coverage.py

Outputs (only):
  /project/community/rmwisene/tts_frontend_analysis/coverage_summary.csv
  /project/community/rmwisene/tts_frontend_analysis/character_summary.json
  /project/community/rmwisene/tts_frontend_analysis/frontend_coverage_report.txt
"""
from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Defaults (Orchard project storage)
# ---------------------------------------------------------------------------
DATA = Path(os.environ.get("DATA", "/project/community/rmwisene/datasets"))
BASELINES = Path(
    os.environ.get("BASELINES", "/project/community/rmwisene/tts_baselines")
)
OUT_DIR = Path(
    os.environ.get(
        "TTS_FRONTEND_ANALYSIS",
        "/project/community/rmwisene/tts_frontend_analysis",
    )
)

KIN_CSV = DATA / "kinyarwanda_tts" / "tts-dataset.csv"
KIN_DIR = DATA / "kinyarwanda_tts"
DAV_DIR = DATA / "kidawida_cv27"
# Prefer frozen cleaned transcripts for frontend checks (source corpus untouched)
DAV_CLEANED_CSV = Path(
    os.environ.get(
        "DAV_CLEANED_CSV",
        "/project/community/rmwisene/tts_data_cleaning/kidawida_cleaned_transcripts.csv",
    )
)
MMS_DIR = BASELINES / "mms_tts_kin"
FLEX_CKPT = BASELINES / "kinya_flex_tts" / "kinya_flex_tts_base_trained.pt"
DEEPKIN_ROOT = Path(
    os.environ.get(
        "DEEPKIN_ROOT",
        "/project/community/rmwisene/code/ac-ai-models/DeepKIN-AgAI",
    )
)

TEXT_KEYS = (
    "text",
    "sentence",
    "transcript",
    "transcription",
    "transcriptions",
    "normalized_text",
    "raw_transcription",
    "raw_text",
    "content",
    "utterance",
    "utterance_text",
    "orthography",
    "phrase",
    "written",
    "prompts",
    "prompt",
    "message",
    "caption",
    "kinyarwanda",
    "rw",
    "kin",
)


# ---------------------------------------------------------------------------
# Text utils
# ---------------------------------------------------------------------------
def normalize_text(s: str) -> str:
    t = unicodedata.normalize("NFC", s or "")
    t = t.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def alphabetic_chars(texts: list[str]) -> set[str]:
    """Unique alphabetic characters (Unicode category starting with L)."""
    out: set[str] = set()
    for t in texts:
        for ch in t:
            if unicodedata.category(ch).startswith("L"):
                out.add(ch)
    return out


def detect_text_column(fieldnames: list[str] | None) -> str | None:
    if not fieldnames:
        return None
    # strip BOM / whitespace on header names
    cleaned = [(f or "").lstrip("\ufeff").strip() for f in fieldnames]
    lower = {f.lower(): orig for f, orig in zip(cleaned, fieldnames) if f}
    for key in TEXT_KEYS:
        if key in lower:
            return lower[key]
    # fuzzy: any column containing these substrings
    for cand in (
        "sentence",
        "transcript",
        "transcription",
        "utterance",
        "prompt",
        "content",
        "text",
    ):
        for low, orig in lower.items():
            if cand in low:
                return orig
    return None


def _sniff_delimiter(sample: str) -> list[str]:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
        return [dialect.delimiter, ",", "\t", ";", "|"]
    except Exception:
        return [",", "\t", ";", "|"]


def is_non_transcript_table(path: Path) -> bool:
    """Skip duration / numeric metadata tables that are not transcripts."""
    n = path.name.lower()
    # Exact / common Common Voice non-transcript tables
    blocked_names = {
        "clip_durations.tsv",
        "clip_durations.csv",
        "sizes.txt",
    }
    if n in blocked_names:
        return True
    blocked_substrings = (
        "clip_duration",
        "durations",
        "duration_ms",
        "file_size",
    )
    return any(s in n for s in blocked_substrings)


def read_table_texts(path: Path, limit: int = 0) -> tuple[list[str], str]:
    """Return (normalized texts, detected column name).

    Only uses columns whose names look like transcript/sentence text.
    Never falls back to numeric columns (e.g. duration).
    """
    if is_non_transcript_table(path):
        raise ValueError(f"Non-transcript table excluded: {path.name}")

    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:8192]
    tried_delims: list[str] = []
    last_fields: list[str] = []

    for delim in _sniff_delimiter(sample):
        if delim in tried_delims:
            continue
        tried_delims.append(delim)
        with path.open(encoding="utf-8-sig", newline="", errors="replace") as f:
            reader = csv.DictReader(f, delimiter=delim)
            fields = list(reader.fieldnames or [])
            if len(fields) <= 1 and delim != ",":
                continue
            last_fields = [((x or "").lstrip("\ufeff").strip()) for x in fields]
            col = detect_text_column(fields)
            # Do NOT score-pick numeric columns — require a transcript-like name
            if col is None:
                continue
            col_l = (col or "").lower()
            if any(
                bad in col_l
                for bad in ("duration", "size", "bytes", "clip", "path", "filename", "age", "votes")
            ) and not any(
                good in col_l for good in ("sentence", "transcript", "text", "utterance", "prompt")
            ):
                continue

        with path.open(encoding="utf-8-sig", newline="", errors="replace") as f:
            reader = csv.DictReader(f, delimiter=delim)
            texts: list[str] = []
            for row in reader:
                raw_cell = (row.get(col) or "").strip()
                if not raw_cell:
                    continue
                # Reject rows that are purely numeric (not transcripts)
                if re.fullmatch(r"[0-9]+([.,][0-9]+)?", raw_cell):
                    continue
                texts.append(normalize_text(raw_cell))
                if limit and len(texts) >= limit:
                    break
            if texts:
                return texts, f"{path.name}:{col} (delim={repr(delim)})"

    raise ValueError(
        f"No transcript/sentence column found in {path}. "
        f"Columns seen: {last_fields or '(none)'}. "
        f"Tried delimiters: {tried_delims}"
    )


def collect_dav_cleaned_texts(
    cleaned_csv: Path, limit: int = 0
) -> tuple[list[str], list[str]]:
    """Load frozen cleaned Kidaw’ida texts (cleaned_text column).

    Keeps all rows including flagged q/x/ä/~ sentences — does not delete them.
    """
    if not cleaned_csv.is_file():
        raise FileNotFoundError(
            f"Missing cleaned Kidaw’ida CSV: {cleaned_csv}\n"
            "Run: python scripts/audit_clean_kidawida_transcripts.py"
        )
    texts: list[str] = []
    notes = [
        f"FROZEN cleaned corpus: {cleaned_csv}",
        "Using column cleaned_text; flagged q/x/ä/~ sentences retained for later review.",
        "Original kidawida_cv27 source files were not modified.",
    ]
    n_flagged = 0
    with cleaned_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "cleaned_text" not in reader.fieldnames:
            raise ValueError(
                f"{cleaned_csv} must have a cleaned_text column; "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            t = normalize_text(row.get("cleaned_text") or "")
            if not t:
                continue
            if (row.get("flagged") or "").strip().lower() in {"yes", "true", "1"}:
                n_flagged += 1
            texts.append(t)
            if limit and len(texts) >= limit:
                break
    notes.append(f"rows_loaded={len(texts)} flagged_rows_included={n_flagged}")
    if not texts:
        raise RuntimeError(f"No cleaned_text rows in {cleaned_csv}")
    return texts, notes


def collect_dav_texts(dav_dir: Path, limit: int = 0) -> tuple[list[str], list[str]]:
    """Collect Kidaw’ida training texts. Prefer train/validated; skip FLEURS."""
    if not dav_dir.is_dir():
        raise FileNotFoundError(f"Missing Kidaw’ida corpus: {dav_dir}")

    candidates: list[Path] = []
    for pat in ("*.tsv", "*.csv"):
        candidates.extend(dav_dir.rglob(pat))

    def rank(p: Path) -> tuple:
        n = p.name.lower()
        score = 100
        if "train" in n:
            score = 0
        elif "validated" in n:
            score = 1
        elif "dev" in n or "test" in n:
            score = 50
        elif any(x in n for x in ("invalidated", "other", "reported")):
            score = 80
        if "fleurs" in str(p).lower() or is_non_transcript_table(p):
            score = 10_000
        return (score, str(p))

    candidates = sorted({c.resolve() for c in candidates}, key=rank)
    notes: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()

    for path in candidates:
        if "fleurs" in str(path).lower():
            notes.append(f"SKIP (FLEURS excluded): {path}")
            continue
        if is_non_transcript_table(path):
            notes.append(f"SKIP (non-transcript table): {path.name}")
            continue
        try:
            part, src = read_table_texts(path, limit=0)
        except Exception as e:
            notes.append(f"skip {path}: {e}")
            continue
        added = 0
        for t in part:
            if t not in seen:
                seen.add(t)
                texts.append(t)
                added += 1
                if limit and len(texts) >= limit:
                    notes.append(f"used {src} (+{added}); hit --limit")
                    return texts, notes
        notes.append(f"used {src} (+{added} unique)")

    if not texts:
        raise RuntimeError(
            f"No Kidaw’ida transcripts found under {dav_dir}. "
            f"Tried: {[str(c) for c in candidates[:20]]}"
        )
    return texts, notes


def read_kin_tts_dataset(path: Path, limit: int = 0) -> tuple[list[str], str]:
    """Parse mbazaNLP-style kinyarwanda tts-dataset.csv.

    Observed formats (no reliable header):
      TTS_1_2 "ntivuga ko nehemiya ..."
      TTS_1_2,ntivuga ko ...
      TTS_1_2,"ntivuga ko ..."
    """
    # First try normal table parse (if someone fixed the CSV later)
    try:
        return read_table_texts(path, limit=limit)
    except ValueError:
        pass

    texts: list[str] = []
    # id + quoted transcript
    pat_quoted = re.compile(r'^\s*\S+\s+"(.*)"\s*$')
    # id,transcript  OR  id,"transcript"
    pat_csv = re.compile(r'^\s*([^,]+),\s*"(.*)"\s*$|^\s*([^,]+),\s*(.*)\s*$')

    with path.open(encoding="utf-8-sig", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            # Skip license-like / header-ish lines with no spoken text
            low = line.lower()
            if low.startswith("license") or low.startswith("http"):
                continue

            text = None
            m = pat_quoted.match(line)
            if m:
                text = m.group(1)
            else:
                m = pat_csv.match(line)
                if m:
                    text = m.group(2) if m.group(2) is not None else m.group(4)

            if text is None:
                # Last resort: if line has a long alphabetic span after first token
                parts = line.split(None, 1)
                if len(parts) == 2 and sum(
                    unicodedata.category(c).startswith("L") for c in parts[1]
                ) >= 8:
                    text = parts[1].strip().strip('"')

            if not text:
                continue
            # Ignore pure ids / numeric
            if re.fullmatch(r"[0-9]+([.,][0-9]+)?", text.strip()):
                continue
            texts.append(normalize_text(text))
            if limit and len(texts) >= limit:
                break

    if not texts:
        raise ValueError(
            f"Could not parse Kinyarwanda transcripts from {path}. "
            "Expected lines like: TTS_1_2 \"...transcript...\""
        )
    return texts, f"{path.name}:line-parser (id + transcript)"


def collect_kin_texts(kin_csv: Path, kin_dir: Path, limit: int = 0) -> tuple[list[str], str]:
    if kin_csv.is_file():
        return read_kin_tts_dataset(kin_csv, limit=limit)
    # fallback: search kin_dir (still not FLEURS)
    for path in sorted(kin_dir.rglob("*.csv")) + sorted(kin_dir.rglob("*.tsv")):
        if "fleurs" in str(path).lower():
            continue
        try:
            if path.name.lower() == "tts-dataset.csv":
                return read_kin_tts_dataset(path, limit=limit)
            return read_table_texts(path, limit=limit)
        except Exception:
            continue
    raise FileNotFoundError(
        f"Missing Kinyarwanda metadata. Expected {kin_csv}"
    )


# ---------------------------------------------------------------------------
# Character comparison
# ---------------------------------------------------------------------------
def character_summary(kin_texts: list[str], dav_texts: list[str]) -> dict:
    kin_alpha = alphabetic_chars(kin_texts)
    dav_alpha = alphabetic_chars(dav_texts)
    shared = kin_alpha & dav_alpha
    kin_only = kin_alpha - dav_alpha
    dav_only = dav_alpha - kin_alpha
    denom = len(kin_alpha | dav_alpha) or 1
    overlap_pct = 100.0 * len(shared) / denom
    # also report coverage of Dav alphabet by Kin alphabet
    dav_covered_by_kin = (
        100.0 * len(dav_alpha & kin_alpha) / (len(dav_alpha) or 1)
    )
    return {
        "n_utterances_kinyarwanda": len(kin_texts),
        "n_utterances_kidawida": len(dav_texts),
        "unique_alphabetic_chars_kinyarwanda": sorted(kin_alpha),
        "unique_alphabetic_chars_kidawida": sorted(dav_alpha),
        "n_unique_alphabetic_kinyarwanda": len(kin_alpha),
        "n_unique_alphabetic_kidawida": len(dav_alpha),
        "shared_alphabetic_chars": sorted(shared),
        "n_shared_alphabetic_chars": len(shared),
        "kinyarwanda_only_alphabetic_chars": sorted(kin_only),
        "kidawida_only_alphabetic_chars": sorted(dav_only),
        "character_overlap_percent_union": round(overlap_pct, 4),
        "kidawida_alphabet_covered_by_kinyarwanda_percent": round(
            dav_covered_by_kin, 4
        ),
        "note": (
            "FLEURS Kinyarwanda was NOT used. Comparison is training corpora only. "
            "This does not prove coverage of the entire Kidaw’ida language."
        ),
    }


# ---------------------------------------------------------------------------
# MMS frontend
# ---------------------------------------------------------------------------
# Punctuation Kin already uses; MMS/VITS normalize typically removes these.
# Their disappearance is expected preprocessing — NOT a compatibility failure.
MMS_EXPECTED_PUNCT = set('.?,!:;"\'“”‘’…—–-_()[]{}')
FOCUS_CHARS = ("x", "ä", "~")  # inspect explicitly for Dav vs Kin/MMS


def mms_frontend_text(tok, text: str) -> str:
    """Apply the tokenizer's own frontend normalize when available.

    VITS/MMS may strip punctuation / change casing. Characters removed by that
    step are intentional frontend behavior, not 'unsupported' vocabulary gaps.
    We do NOT require decode(encode(text)) == text.
    """
    if hasattr(tok, "normalize_text") and callable(tok.normalize_text):
        try:
            if getattr(tok, "normalize", True):
                return tok.normalize_text(text)
        except Exception:
            pass
    return text


def _is_alpha(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L")


def _is_expected_punct(ch: str) -> bool:
    if ch in MMS_EXPECTED_PUNCT:
        return True
    # Unicode punctuation categories (Po, Pd, Pe, …) — intentional strip candidates
    return unicodedata.category(ch).startswith("P")


def _char_token_id(tok, vocab: dict, ch: str):
    """Return tokenizer id for a single character if present, else None."""
    if ch in vocab:
        return vocab[ch]
    # some vocabs only expose via convert_tokens_to_ids
    try:
        tid = tok.convert_tokens_to_ids(ch)
        if tid is not None and tid != tok.unk_token_id:
            return tid
        # if unk_token is the same char (space), treat carefully
        if tid == tok.unk_token_id and ch == tok.unk_token:
            return tid
        if tid is not None and (
            tok.unk_token_id is None or tid != tok.unk_token_id
        ):
            return tid
    except Exception:
        pass
    return None


def inspect_focus_chars(tok, vocab: dict, dav_texts: list[str], kin_charset: set[str]) -> dict:
    """Detailed check for x, ä, ~."""
    out = {}
    for ch in FOCUS_CHARS:
        tid = _char_token_id(tok, vocab, ch)
        in_kin = ch in kin_charset
        # Does normalize preserve the char?
        probe = f"a{ch}b"
        frontend = mms_frontend_text(tok, probe)
        preserved = ch in frontend
        normalized_away = (ch in probe) and (ch not in frontend)

        n_sents = sum(1 for t in dav_texts if ch in t)
        examples = [t for t in dav_texts if ch in t][:5]

        unk_hits = 0
        exc_hits = 0
        for t in (s for s in dav_texts if ch in s):
            try:
                ids = tok(t, return_tensors=None)["input_ids"]
                if tok.unk_token_id is not None and tok.unk_token_id in ids:
                    # Only count if this char likely caused UNK: encode char alone
                    try:
                        one = tok(ch, return_tensors=None)["input_ids"]
                        if tok.unk_token_id in one:
                            unk_hits += 1
                    except Exception:
                        unk_hits += 1
            except Exception:
                exc_hits += 1

        # single-char encode probe
        single_unk = False
        single_exc = None
        single_ids = None
        try:
            single_ids = list(tok(ch, return_tensors=None)["input_ids"])
            if tok.unk_token_id is not None and tok.unk_token_id in single_ids:
                single_unk = True
        except Exception as e:
            single_exc = str(e)

        out[ch] = {
            "in_kinyarwanda_training_corpus": in_kin,
            "tokenizer_id": tid,
            "has_valid_vocab_id": tid is not None and not (
                tok.unk_token_id is not None and tid == tok.unk_token_id and ch != tok.unk_token
            ),
            "preserved_after_normalize": preserved,
            "intentionally_normalized_away": normalized_away,
            "single_char_input_ids": single_ids,
            "single_char_causes_unk": single_unk,
            "single_char_exception": single_exc,
            "dav_sentences_containing": n_sents,
            "dav_sentences_with_unk_when_encoding_char": unk_hits,
            "dav_sentences_with_encode_exception": exc_hits,
            "examples": examples,
        }
    return out


def analyze_mms(
    mms_dir: Path, dav_texts: list[str], kin_texts: list[str]
) -> dict:
    from transformers import AutoTokenizer

    if not mms_dir.is_dir():
        return {
            "model": "MMS-TTS-Kin",
            "ok": False,
            "error": f"Missing MMS model dir: {mms_dir}",
        }

    tok = AutoTokenizer.from_pretrained(str(mms_dir))
    vocab = tok.get_vocab()
    unk_id = tok.unk_token_id
    vocab_chars = {s for s in vocab if isinstance(s, str) and len(s) == 1}

    kin_charset = set("".join(kin_texts))
    # MMS-supported alphabet ≈ vocab char keys ∪ chars attested in Kin after same normalize
    kin_frontend_chars: set[str] = set()
    for t in kin_texts:
        kin_frontend_chars.update(mms_frontend_text(tok, t))
    mms_supported = set(vocab_chars) | {
        c for c in kin_frontend_chars if not c.isspace()
    }

    focus = inspect_focus_chars(tok, vocab, dav_texts, kin_charset)

    ok_sentences = 0
    failed_sentences = 0
    unknown_token_count = 0
    total_tokens = 0
    unsupported_alpha: Counter = Counter()
    unsupported_special: Counter = Counter()
    failures: list[dict] = []
    punct_normalized_away_count = 0  # informational only

    for text in dav_texts:
        frontend = mms_frontend_text(tok, text)

        # Track punctuation removed by normalize (expected — not a failure)
        removed = set(text) - set(frontend)
        if any(_is_expected_punct(c) for c in removed):
            punct_normalized_away_count += 1

        try:
            enc = tok(text, return_tensors=None)
            ids = list(enc["input_ids"])
        except Exception as e:
            failed_sentences += 1
            # true gaps among surviving frontend chars
            for ch in frontend:
                if ch.isspace() or _is_expected_punct(ch):
                    continue
                if ch not in mms_supported and ch not in vocab_chars:
                    if _is_alpha(ch):
                        unsupported_alpha[ch] += 1
                    else:
                        unsupported_special[ch] += 1
            if len(failures) < 10:
                failures.append({"text": text[:160], "reason": f"encode_error: {e}"})
            continue

        total_tokens += len(ids)
        n_unk = 0
        if unk_id is not None:
            n_unk = sum(1 for i in ids if i == unk_id)
        unknown_token_count += n_unk

        # True unsupported = remains AFTER Kin-equivalent MMS normalize,
        # is not expected punctuation, and cannot be mapped by vocab /
        # is a Dav language char absent from Kin+MMS supported alphabet.
        missing_here = []
        for ch in frontend:
            if ch.isspace():
                continue
            # Punctuation still present after normalize: only fail if not in vocab
            # AND not an expected punct that Kin also uses (q/. /, etc. are fine if mapped)
            if _is_expected_punct(ch):
                # If Kin training already has this punct and/or vocab maps it → OK
                if ch in kin_charset or ch in vocab_chars:
                    continue
                # rare punct not in Kin and not in vocab → special unsupported
                if ch not in vocab_chars:
                    unsupported_special[ch] += 1
                    missing_here.append(ch)
                continue

            # Alphabetic / other symbols: must be in MMS vocab to be representable
            in_vocab = ch in vocab_chars
            in_kin = ch in kin_charset
            # Characters already in Kin corpus (e.g. q) are not Dav-novel failures
            # if the tokenizer can map them.
            if in_vocab:
                continue
            # Not in vocab → true unsupported (language gap)
            if _is_alpha(ch):
                unsupported_alpha[ch] += 1
            else:
                unsupported_special[ch] += 1
            missing_here.append(ch)
            # Dav-novel vs Kin is recorded via focus / char summary; still a fail if unmapped

        # Fail only on UNK ids, encode errors (above), or true unmapped non-punct chars.
        # Do NOT fail because punctuation disappeared during normalize.
        bad = bool(n_unk or missing_here)
        if bad:
            failed_sentences += 1
            if len(failures) < 10:
                failures.append(
                    {
                        "text": text[:160],
                        "frontend_text": frontend[:160],
                        "n_unk": n_unk,
                        "true_unsupported_chars": "".join(sorted(set(missing_here)))[:40],
                    }
                )
        else:
            ok_sentences += 1

    n = max(len(dav_texts), 1)
    sentence_cov = 100.0 * ok_sentences / n
    token_cov = 100.0 * (total_tokens - unknown_token_count) / max(total_tokens, 1)
    unsupported_all = sorted(set(unsupported_alpha) | set(unsupported_special))

    # Examples containing x / ä / ~
    focus_examples = {
        ch: focus[ch]["examples"] for ch in FOCUS_CHARS if ch in focus
    }

    return {
        "model": "MMS-TTS-Kin",
        "ok": True,
        "frontend": "transformers.AutoTokenizer / VitsTokenizer",
        "model_dir": str(mms_dir),
        "vocab_size": len(vocab),
        "n_sentences": len(dav_texts),
        "sentence_coverage_percent": round(sentence_cov, 4),
        "token_coverage_percent": round(token_cov, 4),
        "unknown_token_count": unknown_token_count,
        "failed_sentences": failed_sentences,
        "affected_sentence_count": failed_sentences,
        "unsupported_alphabetic_characters": sorted(unsupported_alpha.keys()),
        "unsupported_special_characters": sorted(unsupported_special.keys()),
        "unsupported_kidawida_characters": unsupported_all,
        "unsupported_char_counts": {
            **dict(unsupported_alpha.most_common()),
            **dict(unsupported_special.most_common()),
        },
        "unsupported_alpha_counts": dict(unsupported_alpha.most_common()),
        "unsupported_special_counts": dict(unsupported_special.most_common()),
        "focus_char_inspection": focus,
        "examples_containing_x_ae_tilde": focus_examples,
        "sentences_with_expected_punct_normalized_away": punct_normalized_away_count,
        "example_failures": failures,
        "compatibility": interpret(
            sentence_cov, token_cov, len(unsupported_all), failed_sentences, n
        ),
        "note": (
            "Punctuation intentionally removed by MMS/VITS normalize "
            f"({''.join(sorted(MMS_EXPECTED_PUNCT))}) is NOT treated as unsupported. "
            "Kin training already includes . , ? ! \" : ; and q — those do not auto-fail. "
            "Compatibility focuses on true unmapped chars after the same normalize, "
            "UNK ids, and encode exceptions. Special inspection: x, ä, ~. "
            "decode(encode(text)) round-trip is not required."
        ),
    }


# ---------------------------------------------------------------------------
# Kinya-Flex — discover real frontend from DeepKIN source (no guessing)
# ---------------------------------------------------------------------------
def discover_flex_frontend(deepkin_root: Path) -> dict:
    """Inspect DeepKIN files to identify the real text frontend used by Flex."""
    discovery: dict[str, Any] = {
        "deepkin_root": str(deepkin_root),
        "checkpoint": str(FLEX_CKPT),
        "discovered": False,
        "module": None,
        "function": None,
        "source_file": None,
        "symbol_list_attr": None,
        "symbols": None,
        "evidence": [],
    }
    if not deepkin_root.is_dir():
        discovery["error"] = f"DeepKIN root not found: {deepkin_root}"
        return discovery

    # Evidence from our own eval script + DeepKIN layout
    eval_script = Path(__file__).resolve().parent / "eval_flex_fleurs.py"
    if eval_script.is_file():
        txt = eval_script.read_text(encoding="utf-8", errors="replace")
        if "deepkin.data.kinya_norm import text_to_sequence" in txt:
            discovery["evidence"].append(
                f"{eval_script.name} imports deepkin.data.kinya_norm.text_to_sequence "
                "and calls text_to_sequence(text, norm=True) before FlexKinyaTTS"
            )

    kinya_norm = deepkin_root / "deepkin" / "data" / "kinya_norm.py"
    if kinya_norm.is_file():
        discovery["evidence"].append(f"Found {kinya_norm}")
        src = kinya_norm.read_text(encoding="utf-8", errors="replace")
        if "def text_to_sequence" in src:
            discovery["module"] = "deepkin.data.kinya_norm"
            discovery["function"] = "text_to_sequence"
            discovery["source_file"] = str(kinya_norm)
            discovery["discovered"] = True
            discovery["evidence"].append(
                "kinya_norm.py defines text_to_sequence (Flex text frontend)"
            )

        # Discover symbol tables by reading the file (no guessing names beyond what's present)
        for attr in re.findall(
            r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[\[\(\{]", src, flags=re.M
        ):
            if any(
                k in attr.lower()
                for k in ("symbol", "vocab", "phone", "char", "alphabet")
            ):
                discovery["evidence"].append(
                    f"kinya_norm.py assigns symbol-like name: {attr}"
                )
                if discovery["symbol_list_attr"] is None:
                    discovery["symbol_list_attr"] = attr

        # Also search related modules referenced by import in kinya_norm
        for m in re.findall(r"from\s+([\w.]+)\s+import\s+([^\n]+)", src):
            discovery["evidence"].append(f"kinya_norm imports from {m[0]}: {m[1].strip()}")

    flex_tts = deepkin_root / "deepkin" / "models" / "flex_tts.py"
    if flex_tts.is_file():
        fsrc = flex_tts.read_text(encoding="utf-8", errors="replace")
        discovery["evidence"].append(f"Found {flex_tts}")
        for line in fsrc.splitlines():
            if "text_to_sequence" in line or "kinya_norm" in line:
                discovery["evidence"].append(f"flex_tts.py: {line.strip()[:160]}")

    return discovery


def load_flex_symbols(deepkin_root: Path, discovery: dict) -> list[str] | None:
    if str(deepkin_root) not in sys.path:
        sys.path.insert(0, str(deepkin_root))
    try:
        mod = importlib.import_module("deepkin.data.kinya_norm")
    except Exception:
        return None

    # Prefer attribute named in discovery; else scan module for symbol-like lists/dicts
    candidates: list[str] = []
    if discovery.get("symbol_list_attr"):
        candidates.append(discovery["symbol_list_attr"])
    for name, obj in vars(mod).items():
        if name.startswith("_") and name not in candidates:
            # allow private symbol maps
            pass
        low = name.lower()
        if any(k in low for k in ("symbol", "vocab", "phone", "char", "alphabet")):
            candidates.append(name)

    for name in candidates:
        if not hasattr(mod, name):
            continue
        obj = getattr(mod, name)
        if isinstance(obj, dict):
            return [str(k) for k in obj.keys()]
        if isinstance(obj, (list, tuple, set)):
            return [str(x) for x in obj]
    return None


def analyze_flex(deepkin_root: Path, dav_texts: list[str]) -> dict:
    discovery = discover_flex_frontend(deepkin_root)
    result: dict[str, Any] = {
        "model": "Kinya-Flex-TTS",
        "checkpoint": str(FLEX_CKPT),
        "checkpoint_exists": FLEX_CKPT.is_file(),
        "frontend_discovery": discovery,
    }

    if not discovery.get("discovered"):
        result["ok"] = False
        result["error"] = (
            "Could not identify Flex text frontend from local DeepKIN files. "
            "Expected deepkin/data/kinya_norm.py::text_to_sequence "
            f"under {deepkin_root}."
        )
        result["compatibility"] = "Significant adaptation needed"
        # still try symbol comparison if any file-level list was found later
        return result

    if str(deepkin_root) not in sys.path:
        sys.path.insert(0, str(deepkin_root))

    try:
        mod = importlib.import_module(discovery["module"])
        text_to_sequence = getattr(mod, discovery["function"])
    except Exception as e:
        result["ok"] = False
        result["error"] = f"Failed to import discovered frontend: {e}"
        result["compatibility"] = "Significant adaptation needed"
        return result

    # Confirm signature from source, not guess
    try:
        sig = str(inspect.signature(text_to_sequence))
        result["frontend_signature"] = f"{discovery['module']}.{discovery['function']}{sig}"
    except Exception:
        result["frontend_signature"] = (
            f"{discovery['module']}.{discovery['function']}(...)"
        )

    symbols = load_flex_symbols(deepkin_root, discovery)
    symbol_chars = set()
    if symbols:
        for s in symbols:
            if len(s) == 1:
                symbol_chars.add(s.lower())
        result["discovered_symbol_count"] = len(symbols)
        result["discovered_symbol_list_attr"] = discovery.get("symbol_list_attr")

    ok_sentences = 0
    failed_sentences = 0
    unknown_token_count = 0
    total_symbols = 0
    unsupported_chars: Counter = Counter()
    failures: list[dict] = []

    for text in dav_texts:
        try:
            # eval_flex_fleurs.py uses norm=True — match that call site
            seq = text_to_sequence(text, norm=True)
        except TypeError:
            try:
                seq = text_to_sequence(text)
            except Exception as e:
                failed_sentences += 1
                if len(failures) < 10:
                    failures.append({"text": text[:160], "reason": f"encode_error: {e}"})
                continue
        except Exception as e:
            failed_sentences += 1
            if len(failures) < 10:
                failures.append({"text": text[:160], "reason": f"encode_error: {e}"})
            continue

        if seq is None:
            failed_sentences += 1
            if len(failures) < 10:
                failures.append({"text": text[:160], "reason": "sequence_is_None"})
            continue

        try:
            seq_list = list(seq)
        except TypeError:
            failed_sentences += 1
            if len(failures) < 10:
                failures.append({"text": text[:160], "reason": "sequence_not_iterable"})
            continue

        if len(seq_list) == 0 and text.strip():
            failed_sentences += 1
            if len(failures) < 10:
                failures.append({"text": text[:160], "reason": "empty_sequence"})
            continue

        total_symbols += len(seq_list)

        # If we have a char-level symbol inventory, count unsupported alpha chars
        missing = []
        if symbol_chars:
            for ch in text:
                if not unicodedata.category(ch).startswith("L"):
                    continue
                if ch not in symbol_chars:
                    unsupported_chars[ch] += 1
                    missing.append(ch)
                    unknown_token_count += 1

        if missing:
            failed_sentences += 1
            if len(failures) < 10:
                failures.append(
                    {
                        "text": text[:160],
                        "missing_chars": "".join(sorted(set(missing)))[:40],
                    }
                )
        else:
            ok_sentences += 1

    n = max(len(dav_texts), 1)
    sentence_cov = 100.0 * ok_sentences / n
    if symbol_chars and total_symbols:
        # symbol coverage ≈ non-missing alpha over (approx) — use token proxy
        # Here unknown_token_count counted missing alpha occurrences
        alpha_total = sum(
            1 for t in dav_texts for ch in t if unicodedata.category(ch).startswith("L")
        )
        token_cov = 100.0 * (alpha_total - unknown_token_count) / max(alpha_total, 1)
    elif total_symbols:
        # No inventory: treat successful encode as full token coverage
        token_cov = 100.0 * (n - failed_sentences) / n
        # unknown unknown without inventory
    else:
        token_cov = 0.0

    unsupported = sorted(unsupported_chars.keys())
    result.update(
        {
            "ok": True,
            "frontend": result.get("frontend_signature"),
            "n_sentences": len(dav_texts),
            "sentence_coverage_percent": round(sentence_cov, 4),
            "token_coverage_percent": round(token_cov, 4),
            "unknown_token_count": unknown_token_count,
            "failed_sentences": failed_sentences,
            "unsupported_kidawida_characters": unsupported,
            "unsupported_char_counts": dict(unsupported_chars.most_common()),
            "example_failures": failures,
            "compatibility": interpret(
                sentence_cov, token_cov, len(unsupported), failed_sentences, n
            ),
            "note": (
                "Frontend identified from DeepKIN source + eval_flex_fleurs.py call site "
                "(text_to_sequence). Checkpoint provides weights; text frontend is code-side."
            ),
        }
    )
    if not symbol_chars:
        result["symbol_inventory_note"] = (
            "Ran text_to_sequence on all Dav sentences. "
            "No explicit character/symbol list attribute was loadable from kinya_norm; "
            "unsupported-character list may be incomplete (failures still counted)."
        )
    return result


def interpret(
    sentence_cov: float,
    token_cov: float,
    n_unsupported: int,
    failed: int,
    n: int,
) -> str:
    """Final label for one baseline on *this corpus* only."""
    fail_rate = failed / max(n, 1)
    if sentence_cov >= 99.0 and token_cov >= 99.0 and n_unsupported == 0:
        return "Full compatibility"
    if sentence_cov >= 90.0 and token_cov >= 95.0 and fail_rate <= 0.10:
        return "Minor frontend adaptation needed"
    return "Significant adaptation needed"


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def write_coverage_csv(path: Path, mms: dict, flex: dict) -> None:
    rows = []
    for model in (mms, flex):
        name = model.get("model", "?")
        if not model.get("ok"):
            rows.append(
                {
                    "Model": name,
                    "Sentence Coverage": "",
                    "Token/Symbol Coverage": "",
                    "Unknown Count": "",
                    "Failed Sentences": "",
                    "Unsupported Kidawida Characters": model.get("error", "N/A"),
                }
            )
            continue
        unsup = model.get("unsupported_kidawida_characters") or []
        if model.get("model") == "MMS-TTS-Kin":
            # Prefer split alpha/special in the CSV cell for clarity
            alpha = "".join(model.get("unsupported_alphabetic_characters") or [])
            special = "".join(model.get("unsupported_special_characters") or [])
            unsup_cell = f"alpha=[{alpha}] special=[{special}]"
        else:
            unsup_cell = "".join(unsup)
        rows.append(
            {
                "Model": name,
                "Sentence Coverage": model.get("sentence_coverage_percent", ""),
                "Token/Symbol Coverage": model.get("token_coverage_percent", ""),
                "Unknown Count": model.get("unknown_token_count", ""),
                "Failed Sentences": model.get(
                    "affected_sentence_count", model.get("failed_sentences", "")
                ),
                "Unsupported Kidawida Characters": unsup_cell,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "Model",
                "Sentence Coverage",
                "Token/Symbol Coverage",
                "Unknown Count",
                "Failed Sentences",
                "Unsupported Kidawida Characters",
            ],
        )
        w.writeheader()
        w.writerows(rows)


def write_report(
    path: Path,
    char_sum: dict,
    mms: dict,
    flex: dict,
    kin_src: str,
    dav_notes: list[str],
) -> None:
    lines: list[str] = []
    lines += [
        "TTS frontend coverage report",
        "============================",
        "",
        "Scope: Kinyarwanda + Kidaw’ida TRAINING corpora only.",
        "FLEURS Kinyarwanda was used for baseline evaluation only and is EXCLUDED here.",
        "This does NOT claim support for the entire Kidaw’ida language — only the available training corpus.",
        "",
        f"Kinyarwanda source: {kin_src}",
        f"Kidaw’ida dir: {DAV_DIR}",
        "",
        "Kidaw’ida file selection notes:",
    ]
    for n in dav_notes[:40]:
        lines.append(f"  - {n}")
    lines += [
        "",
        "1) Corpus character comparison (alphabetic)",
        "------------------------------------------",
        f"  Kin utterances: {char_sum['n_utterances_kinyarwanda']}",
        f"  Dav utterances: {char_sum['n_utterances_kidawida']}",
        f"  Kin unique alpha chars ({char_sum['n_unique_alphabetic_kinyarwanda']}): "
        f"{''.join(char_sum['unique_alphabetic_chars_kinyarwanda'])}",
        f"  Dav unique alpha chars ({char_sum['n_unique_alphabetic_kidawida']}): "
        f"{''.join(char_sum['unique_alphabetic_chars_kidawida'])}",
        f"  Shared ({char_sum['n_shared_alphabetic_chars']}): "
        f"{''.join(char_sum['shared_alphabetic_chars'])}",
        f"  Kin-only: {''.join(char_sum['kinyarwanda_only_alphabetic_chars'])}",
        f"  Dav-only: {''.join(char_sum['kidawida_only_alphabetic_chars'])}",
        f"  Overlap % (shared / union): {char_sum['character_overlap_percent_union']}",
        f"  Dav alphabet covered by Kin %: "
        f"{char_sum['kidawida_alphabet_covered_by_kinyarwanda_percent']}",
        "",
        "2) MMS-TTS-Kin frontend",
        "----------------------",
        "  Criteria: true unsupported chars AFTER MMS normalize (same as Kin), UNK ids,",
        "  encode exceptions. Expected punctuation removal (. , ? ! \" : ; …) is NOT a failure.",
        "  Kin already has those punctuations and q — they do not auto-fail.",
        "  NOT used: decode(encode(text)) round-trip identity.",
    ]

    lines.extend(_format_model_block(mms))
    if mms.get("ok") and mms.get("focus_char_inspection"):
        lines += ["", "  Focus inspection (x, ä, ~):"]
        for ch, info in mms["focus_char_inspection"].items():
            lines.append(
                f"    '{ch}': in_kin={info.get('in_kinyarwanda_training_corpus')} "
                f"vocab_id={info.get('tokenizer_id')} "
                f"valid_id={info.get('has_valid_vocab_id')} "
                f"preserved={info.get('preserved_after_normalize')} "
                f"normalized_away={info.get('intentionally_normalized_away')} "
                f"single_unk={info.get('single_char_causes_unk')} "
                f"dav_sents={info.get('dav_sentences_containing')}"
            )
            for ex in (info.get("examples") or [])[:3]:
                lines.append(f"      e.g. {ex[:160]}")

    lines += ["", "3) Kinya-Flex-TTS frontend", "--------------------------"]
    lines.extend(_format_model_block(flex))
    if flex.get("frontend_discovery"):
        lines += ["", "  Flex frontend discovery evidence:"]
        for ev in flex["frontend_discovery"].get("evidence", [])[:20]:
            lines.append(f"    - {ev}")

    lines += [
        "",
        "4) Final decision (this training corpus only)",
        "---------------------------------------------",
        f"  MMS-TTS-Kin:     {mms.get('compatibility', mms.get('error', 'N/A'))}",
        f"  Kinya-Flex-TTS:  {flex.get('compatibility', flex.get('error', 'N/A'))}",
        "",
        "Labels:",
        "  - Full compatibility",
        "  - Minor frontend adaptation needed",
        "  - Significant adaptation needed",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_model_block(model: dict) -> list[str]:
    if not model.get("ok"):
        return [
            f"  ERROR: {model.get('error', 'unknown')}",
            f"  Compatibility: {model.get('compatibility', 'Significant adaptation needed')}",
        ]
    lines = [
        f"  Frontend: {model.get('frontend')}",
        f"  Sentence coverage %: {model.get('sentence_coverage_percent')}",
        f"  Token/symbol coverage %: {model.get('token_coverage_percent')}",
        f"  Unknown token count: {model.get('unknown_token_count')}",
        f"  Failed / affected sentences: {model.get('failed_sentences')}",
        f"  Unsupported alphabetic: "
        f"{''.join(model.get('unsupported_alphabetic_characters') or []) or '(none)'}",
        f"  Unsupported special: "
        f"{''.join(model.get('unsupported_special_characters') or []) or '(none)'}",
        f"  Unsupported (all): "
        f"{''.join(model.get('unsupported_kidawida_characters') or []) or '(none)'}",
        f"  Compatibility: {model.get('compatibility')}",
        "  Example failures (up to 10):",
    ]
    fails = model.get("example_failures") or []
    if not fails:
        lines.append("    (none)")
    for i, ex in enumerate(fails, 1):
        lines.append(f"    [{i}] {json.dumps(ex, ensure_ascii=False)}")
    if model.get("symbol_inventory_note"):
        lines.append(f"  Note: {model['symbol_inventory_note']}")
    if model.get("note"):
        lines.append(f"  Note: {model['note']}")
    return lines


def print_summary_table(mms: dict, flex: dict) -> None:
    def cell(m: dict, key: str) -> str:
        if not m.get("ok"):
            if key == "Unsupported Kidawida Characters":
                return str(m.get("error", "N/A"))[:60]
            return "N/A"
        if key == "Model":
            return str(m.get("model"))
        if key == "Sentence Coverage":
            return f"{m.get('sentence_coverage_percent')}%"
        if key == "Token/Symbol Coverage":
            return f"{m.get('token_coverage_percent')}%"
        if key == "Unknown Count":
            return str(m.get("unknown_token_count"))
        if key == "Failed Sentences":
            return str(m.get("failed_sentences"))
        if key == "Unsupported Kidawida Characters":
            return "".join(m.get("unsupported_kidawida_characters") or []) or "(none)"
        return ""

    cols = [
        "Model",
        "Sentence Coverage",
        "Token/Symbol Coverage",
        "Unknown Count",
        "Failed Sentences",
        "Unsupported Kidawida Characters",
    ]
    rows = [[cell(m, c) for c in cols] for m in (mms, flex)]
    widths = [max(len(cols[i]), *(len(r[i]) for r in rows)) for i in range(len(cols))]

    def fmt(row: list[str]) -> str:
        return " | ".join(row[i].ljust(widths[i]) for i in range(len(cols)))

    sep = "-+-".join("-" * w for w in widths)
    print("\n=== coverage_summary ===")
    print(fmt(cols))
    print(sep)
    for r in rows:
        print(fmt(r))
    print()
    print("Compatibility:")
    print(f"  MMS-TTS-Kin:    {mms.get('compatibility', mms.get('error'))}")
    print(f"  Kinya-Flex-TTS: {flex.get('compatibility', flex.get('error'))}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Kin/Dav training-corpus frontend coverage for MMS + Flex"
    )
    ap.add_argument("--kin_csv", type=Path, default=KIN_CSV)
    ap.add_argument("--kin_dir", type=Path, default=KIN_DIR)
    ap.add_argument("--dav_dir", type=Path, default=DAV_DIR)
    ap.add_argument(
        "--dav_cleaned_csv",
        type=Path,
        default=DAV_CLEANED_CSV,
        help="Frozen cleaned Kidaw’ida transcripts CSV (default). "
        "Use --use_raw_dav to read kidawida_cv27 instead.",
    )
    ap.add_argument(
        "--use_raw_dav",
        action="store_true",
        help="Load raw kidawida_cv27 instead of the frozen cleaned CSV",
    )
    ap.add_argument("--mms_dir", type=Path, default=MMS_DIR)
    ap.add_argument("--deepkin_root", type=Path, default=DEEPKIN_ROOT)
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=0, help="Max utterances/language (0=all)")
    ap.add_argument("--skip_mms", action="store_true")
    ap.add_argument("--skip_flex", action="store_true")
    args = ap.parse_args()

    print("Loading Kinyarwanda training texts (NOT FLEURS)…")
    kin_texts, kin_src = collect_kin_texts(args.kin_csv, args.kin_dir, args.limit)
    print(f"  n={len(kin_texts)} from {kin_src}")

    if args.use_raw_dav:
        print("Loading Kidaw’ida RAW training texts…")
        dav_texts, dav_notes = collect_dav_texts(args.dav_dir, args.limit)
    else:
        print("Loading Kidaw’ida FROZEN cleaned transcripts…")
        dav_texts, dav_notes = collect_dav_cleaned_texts(
            args.dav_cleaned_csv, args.limit
        )
    print(f"  n={len(dav_texts)}")
    for note in dav_notes[:5]:
        print(f"  · {note}")

    char_sum = character_summary(kin_texts, dav_texts)

    if args.skip_mms:
        mms: dict = {"model": "MMS-TTS-Kin", "ok": False, "error": "skipped"}
    else:
        print("Checking MMS-TTS-Kin frontend…")
        mms = analyze_mms(args.mms_dir, dav_texts, kin_texts)

    if args.skip_flex:
        flex: dict = {"model": "Kinya-Flex-TTS", "ok": False, "error": "skipped"}
    else:
        print("Discovering + checking Kinya-Flex frontend…")
        flex = analyze_flex(args.deepkin_root, dav_texts)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "coverage_summary.csv"
    json_path = out / "character_summary.json"
    txt_path = out / "frontend_coverage_report.txt"

    write_coverage_csv(csv_path, mms, flex)
    json_path.write_text(
        json.dumps(char_sum, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_report(txt_path, char_sum, mms, flex, kin_src, dav_notes)

    print_summary_table(mms, flex)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
