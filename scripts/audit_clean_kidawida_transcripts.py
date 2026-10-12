#!/usr/bin/env python3
"""Audit + safely clean Kidaw’ida transcripts (no source overwrite).

Loads transcript/sentence fields from kidawida_cv27, audits suspicious symbols
(_, ~, q, x, ä), applies only safe automatic cleaning, and writes a cleaned
copy plus reports under /project/community/rmwisene/tts_data_cleaning/.

  python scripts/audit_clean_kidawida_transcripts.py

Does NOT claim q/x/ä are invalid Kidaw’ida.
Does NOT delete sentences containing those characters.
Does NOT overwrite any source dataset file.
"""
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


DAV_DIR = Path("/project/community/rmwisene/datasets/kidawida_cv27")
OUT_DIR = Path("/project/community/rmwisene/tts_data_cleaning")

TEXT_KEYS = (
    "sentence",
    "text",
    "transcript",
    "transcription",
    "normalized_text",
    "raw_transcription",
    "utterance",
    "prompt",
    "content",
)

# Ordinary punctuation — normalize safely; not treated as incompatibility.
NORMAL_PUNCT = set('.?,!:;"')

# Suspicious symbols to audit (never silently replace q/x/ä/~)
SUSPICIOUS = ("_", "~", "q", "x", "ä")

# Unicode punctuation / quote variants → ASCII-ish safe forms
QUOTE_MAP = {
    "\u2018": "'",  # ‘
    "\u2019": "'",  # ’
    "\u201A": "'",  # ‚
    "\u201B": "'",  # ‛
    "\u2032": "'",  # ′
    "\u201C": '"',  # “
    "\u201D": '"',  # ”
    "\u201E": '"',  # „
    "\u2033": '"',  # ″
    "\u00AB": '"',  # «
    "\u00BB": '"',  # »
    "\u2013": "-",  # –
    "\u2014": "-",  # —
    "\u2212": "-",  # −
    "\u2026": "...",  # …
    "\u00A0": " ",  # nbsp
    "\u2009": " ",
    "\u200A": " ",
    "\u202F": " ",
}


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def detect_text_column(fieldnames: list[str] | None) -> str | None:
    if not fieldnames:
        return None
    cleaned = [(f or "").lstrip("\ufeff").strip() for f in fieldnames]
    lower = {f.lower(): orig for f, orig in zip(cleaned, fieldnames) if f}
    for key in TEXT_KEYS:
        if key in lower:
            return lower[key]
    for cand in ("sentence", "transcript", "transcription", "utterance", "text"):
        for low, orig in lower.items():
            if cand in low:
                return orig
    return None


def is_non_transcript_table(path: Path) -> bool:
    n = path.name.lower()
    if n in {"clip_durations.tsv", "clip_durations.csv", "sizes.txt"}:
        return True
    return any(
        s in n
        for s in ("clip_duration", "durations", "duration_ms", "file_size")
    )


def load_kidawida_rows(dav_dir: Path) -> list[dict]:
    """Load unique transcript rows with source_file + original_text."""
    if not dav_dir.is_dir():
        raise FileNotFoundError(f"Missing Kidaw’ida corpus: {dav_dir}")

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    candidates = sorted(
        {p.resolve() for p in list(dav_dir.rglob("*.tsv")) + list(dav_dir.rglob("*.csv"))},
        key=lambda p: (
            0
            if "train" in p.name.lower()
            else 1
            if "validated" in p.name.lower()
            else 2,
            str(p),
        ),
    )

    for path in candidates:
        if "fleurs" in str(path).lower() or is_non_transcript_table(path):
            continue
        # sniff delimiter
        sample = path.read_text(encoding="utf-8-sig", errors="replace")[:4096]
        try:
            delim = csv.Sniffer().sniff(sample, delimiters="\t,;|").delimiter
        except Exception:
            delim = "\t" if path.suffix.lower() == ".tsv" else ","

        with path.open(encoding="utf-8-sig", newline="", errors="replace") as f:
            reader = csv.DictReader(f, delimiter=delim)
            col = detect_text_column(list(reader.fieldnames or []))
            if col is None:
                continue
            col_l = col.lower()
            if any(
                bad in col_l
                for bad in ("duration", "size", "bytes", "path", "filename")
            ) and not any(
                good in col_l for good in ("sentence", "transcript", "text", "utterance")
            ):
                continue

            rel = str(path.relative_to(dav_dir)) if path.is_relative_to(dav_dir) else str(path)
            for row in reader:
                raw = (row.get(col) or "").strip()
                if not raw:
                    continue
                if re.fullmatch(r"[0-9]+([.,][0-9]+)?", raw):
                    continue
                key = (rel, raw)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"source_file": rel, "original_text": raw})

    if not rows:
        raise RuntimeError(f"No Kidaw’ida transcripts found under {dav_dir}")
    return rows


# ---------------------------------------------------------------------------
# Safe automatic cleaning (no silent q/x/ä/~ replacement)
# ---------------------------------------------------------------------------
CLEANING_RULES = [
    "Unicode NFC normalization",
    "Map Unicode quote/dash/ellipsis/nbsp variants to ASCII ' \" - ... space",
    "Fix unambiguous malformed '; ' used as word separator (w;ose → w'ose) only when letter;letter",
    "Collapse repeated underscores (____ → _)",
    "Remove standalone underscore tokens and trailing/leading underscore placeholders",
    "Remove underscores that are only fillers between spaces",
    "Collapse repeated whitespace; strip ends",
    "Lowercase for consistency with TTS pipeline normalize (preserves ä)",
    "Do NOT replace/remove q, x, ä, or ~ automatically",
]


def safe_clean(text: str) -> tuple[str, list[str]]:
    """Return (cleaned_text, list of rule tags applied)."""
    applied: list[str] = []
    t = nfc(text)
    if t != text:
        applied.append("nfc")

    # Unicode punctuation / space variants
    out = []
    mapped = False
    for ch in t:
        if ch in QUOTE_MAP:
            out.append(QUOTE_MAP[ch])
            mapped = True
        else:
            out.append(ch)
    t2 = "".join(out)
    if mapped:
        applied.append("unicode_punct_map")
        t = t2

    # letter;letter → letter'letter (annotation corruption like w;ose)
    t2, n = re.subn(r"(?<=\w);(?=\w)", "'", t)
    if n:
        applied.append("semicolon_to_apostrophe_between_letters")
        t = t2

    # Collapse repeated underscores
    t2, n = re.subn(r"_{2,}", "_", t)
    if n:
        applied.append("collapse_repeated_underscores")
        t = t2

    # Remove underscore-only placeholders / trailing fillers
    t2 = re.sub(r"(^|\s)_+(\s|$)", r"\1\2", t)
    t2 = re.sub(r"_+$", "", t2)
    t2 = re.sub(r"^_+", "", t2)
    if t2 != t:
        applied.append("remove_underscore_placeholders")
        t = t2

    # Whitespace
    t2 = re.sub(r"\s+", " ", t).strip()
    if t2 != t:
        applied.append("whitespace_collapse")
        t = t2

    # Lowercase (ä preserved under NFC lower)
    t2 = t.lower()
    if t2 != t:
        applied.append("lowercase")
        t = t2

    return t, applied


def classify_symbol(symbol: str, examples: list[str]) -> str:
    """Heuristic category — not a linguistic claim of invalidity."""
    if symbol == "_":
        return "a) likely punctuation/annotation noise"
    if symbol == "~":
        # often looks like corrupted apostrophe / spacing (w~adacha, raw~o)
        return "b) likely typo/corruption"
    if symbol == "q":
        # may be loan/code-switch or OCR; do not auto-invalidate
        return "c) possible loanword/code-switching"
    if symbol == "x":
        # appears in 'text' loanword example — possible code-switch; could also be genuine
        joined = " ".join(examples[:10]).lower()
        if re.search(r"\btext\b|\bex\b|xbox|twitter|http", joined):
            return "c) possible loanword/code-switching"
        return "d) possible genuine Kidaw’ida character"
    if symbol == "ä":
        return "d) possible genuine Kidaw’ida character"
    return "d) possible genuine Kidaw’ida character"


def flag_row(cleaned: str) -> tuple[bool, str, str]:
    """Return (flagged, flagged_symbols, review_reason)."""
    found = [s for s in SUSPICIOUS if s in cleaned]
    # also flag if original cleaning left underscore mid-token oddly — still keep
    if not found:
        return False, "", ""
    reasons = []
    for s in found:
        if s == "_":
            reasons.append("_ residual placeholder/noise — manual check")
        elif s == "~":
            reasons.append("~ may be corrupted apostrophe/spacing — manual check")
        elif s == "q":
            reasons.append("q kept; possible loan/typo — do not auto-delete — manual check")
        elif s == "x":
            reasons.append("x kept; possible loanword or genuine — manual check")
        elif s == "ä":
            reasons.append("ä kept; possible genuine orthography — manual check")
    return True, " ".join(found), "; ".join(reasons)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dav_dir", type=Path, default=DAV_DIR)
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print(f"Loading Kidaw’ida transcripts from {args.dav_dir} …")
    rows = load_kidawida_rows(args.dav_dir)
    if args.limit:
        rows = rows[: args.limit]
    print(f"  n={len(rows)}")

    # --- Audit on original text (before cleaning) ---
    # Collect per-symbol stats with examples + sources
    sym_occ: Counter = Counter()
    sym_sents: dict[str, set[str]] = {s: set() for s in SUSPICIOUS}
    sym_examples: dict[str, list[tuple[str, str]]] = defaultdict(list)  # (source, text)

    for row in rows:
        t = nfc(row["original_text"]).lower()
        for s in SUSPICIOUS:
            if s in t:
                c = t.count(s)
                sym_occ[s] += c
                sym_sents[s].add(t)
                if len(sym_examples[s]) < 20:
                    sym_examples[s].append((row["source_file"], row["original_text"]))

    # --- Clean ---
    cleaned_rows: list[dict] = []
    for row in rows:
        original = row["original_text"]
        cleaned, _applied = safe_clean(original)
        # Traceability: changed if cleaned string differs from the source string
        changed_flag = cleaned != original
        flagged, flagged_symbols, review_reason = flag_row(cleaned)

        cleaned_rows.append(
            {
                "source_file": row["source_file"],
                "original_text": original,
                "cleaned_text": cleaned,
                "changed": "yes" if changed_flag else "no",
                "flagged": "yes" if flagged else "no",
                "flagged_symbols": flagged_symbols,
                "review_reason": review_reason,
            }
        )

    n_changed = sum(1 for r in cleaned_rows if r["changed"] == "yes")
    n_unchanged = sum(1 for r in cleaned_rows if r["changed"] == "no")
    n_flagged = sum(1 for r in cleaned_rows if r["flagged"] == "yes")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # 1) cleaned transcripts
    cleaned_path = out / "kidawida_cleaned_transcripts.csv"
    with cleaned_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "source_file",
                "original_text",
                "cleaned_text",
                "changed",
                "flagged",
                "flagged_symbols",
                "review_reason",
            ],
        )
        w.writeheader()
        w.writerows(cleaned_rows)

    # 2) suspicious symbol report
    sus_path = out / "suspicious_symbol_report.csv"
    with sus_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "symbol",
                "occurrence_count",
                "sentence_count",
                "likely_category",
                "example_sentences",
            ],
        )
        w.writeheader()
        for s in SUSPICIOUS:
            examples = sym_examples[s]
            ex_texts = [t for _, t in examples]
            cat = classify_symbol(s, ex_texts)
            # pack examples with source
            packed = " || ".join(
                f"[{src}] {txt}" for src, txt in examples[:20]
            )
            w.writerow(
                {
                    "symbol": s,
                    "occurrence_count": sym_occ[s],
                    "sentence_count": len(sym_sents[s]),
                    "likely_category": cat,
                    "example_sentences": packed,
                }
            )

    # 3) cleaning_report.txt
    report_path = out / "cleaning_report.txt"
    lines = [
        "Kidaw’ida transcript cleaning audit",
        "===================================",
        "",
        f"Source corpus: {args.dav_dir}",
        f"Output dir:    {out}",
        "Original source files were NOT modified.",
        "",
        f"Total sentences processed: {len(cleaned_rows)}",
        f"Automatically cleaned (changed=yes): {n_changed}",
        f"Unchanged: {n_unchanged}",
        f"Flagged for manual review: {n_flagged}",
        "",
        "Suspicious symbol counts (on original text, before cleaning):",
        f"  _ occurrences: {sym_occ['_']}  (sentences: {len(sym_sents['_'])})",
        f"  ~ occurrences: {sym_occ['~']}  (sentences: {len(sym_sents['~'])})",
        f"  q occurrences: {sym_occ['q']}  (sentences: {len(sym_sents['q'])})",
        f"  x occurrences: {sym_occ['x']}  (sentences: {len(sym_sents['x'])})",
        f"  ä occurrences: {sym_occ['ä']}  (sentences: {len(sym_sents['ä'])})",
        "",
        "Likely categories (heuristic; not linguistic judgments of invalidity):",
    ]
    for s in SUSPICIOUS:
        ex_texts = [t for _, t in sym_examples[s]]
        lines.append(f"  '{s}': {classify_symbol(s, ex_texts)}")
    lines += [
        "",
        "Exact automatic cleaning rules applied:",
    ]
    for i, rule in enumerate(CLEANING_RULES, 1):
        lines.append(f"  {i}. {rule}")
    lines += [
        "",
        "Ordinary punctuation (. , ? ! \" : ;) is normalized where needed and is",
        "NOT treated as tokenizer/frontend incompatibility by itself.",
        "",
        "Recommendations before training / before rerunning frontend coverage:",
        "  - Manually review sentences flagged with ~ (likely corruption of apostrophe).",
        "  - Manually review residual _ placeholders.",
        "  - Inspect q / x / ä examples; do NOT auto-delete;",
        "    decide per-item: keep as genuine, treat as loanword, or correct typo.",
        "  - Then rerun analyze_frontend_coverage.py on cleaned_text.",
        "",
        "Disclaimer: This audit does not claim q, x, or ä are invalid Kidaw’ida.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Terminal summary
    print()
    print(f"Total sentences: {len(cleaned_rows)}")
    print(f"Automatically cleaned: {n_changed}")
    print(f"Flagged for review: {n_flagged}")
    print(f"_ occurrences: {sym_occ['_']}")
    print(f"~ occurrences: {sym_occ['~']}")
    print(f"q occurrences: {sym_occ['q']}")
    print(f"x occurrences: {sym_occ['x']}")
    print(f"ä occurrences: {sym_occ['ä']}")
    print()
    print(f"Wrote {cleaned_path}")
    print(f"Wrote {sus_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
