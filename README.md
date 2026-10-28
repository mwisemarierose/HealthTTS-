# HealthTTS

Standalone TTS project (separate from HealthASR). Bilingual TTS for **Kinyarwanda** + **Kidaw’ida**.

This repo is **code only**. Datasets are not included — create them on shared storage when you set up a machine (e.g. Orchard).

## Repo layout

```text
HealthTTS/
├── configs/
├── training/
├── eval/
├── baselines/
├── scripts/
└── README.md
```

## Clone (Orchard)

```bash
cd ~
git clone https://github.com/mwisemarierose/HealthTTS-.git HealthTTS
cd HealthTTS
```

## Create dataset folders (required before download)

Data lives under project storage, **not** in this repo / not under home:

```bash
export DATA=/project/community/rmwisene/datasets
mkdir -p "$DATA"/{kinyarwanda_tts,afrispeak_kinyarwanda_female,afrispeak_kinyarwanda_male,kidawida_cv27,fleurs_kinyarwanda}
# or: bash scripts/setup_orchard_dirs.sh
```

Expected data layout after downloads:

```text
/project/community/rmwisene/datasets/
├── kinyarwanda_tts/                   # mbaza TTS (HF: mbazaNLP/kinyarwanda-tts-dataset)
├── afrispeak_kinyarwanda_female/      # AfriSpeak Female TTS
├── afrispeak_kinyarwanda_male/        # AfriSpeak Male TTS (gated — accept on HF)
├── kidawida_cv27/                     # train/adapt (Mozilla Data Collective)
└── fleurs_kinyarwanda/                # baseline eval only (HF: mbazaNLP/fleurs-kinyarwanda)
```

## Data roles (fixed)

| Dataset | Role | Notes |
|---------|------|--------|
| `kinyarwanda_tts` + AfriSpeak F/M | **Train / adapt** (Kin pool) | Merge via `prepare_kinyarwanda_tts_corpus.py` (~20 h); one shared train/dev/test |
| `kidawida_cv27` / ASR expanded | **Train / adapt** (Dav) | Prepare via `prepare_kidawida_tts_corpus.py` |
| `fleurs_kinyarwanda` | **Baseline evaluation only** (Kin) | Do **not** train or adapt on this set |

Nothing changes: **FLEURS stays baseline evaluation** for Kinyarwanda.

## Experiments

All fine-tunes init from the **same** pretrained B0/B1 weights (not sequential E1→E3).  
Kin train hours = combined pool `train.csv` (**17.89 h**). Kidaw’ida **27.07 h** = ASR expanded **train** only.  
Balanced bilingual (E3/E4): sample a reproducible **~17.89 h** subset of Kidaw’ida train (`seed` logged).

| Exp. | Model | Training condition | Kinyarwanda train | Kidaw’ida train | Data balance | Epochs | Main question |
|------|-------|--------------------|------------------:|----------------:|--------------|-------:|---------------|
| **B0** | MMS-TTS-Kin | Pretrained baseline, no fine-tuning | 0 h | 0 h | — | 0 | How well does the Kinyarwanda-pretrained baseline perform on Kinyarwanda, and what is its **zero-shot Kidaw’ida** capability? |
| **B1** | Kinya-Flex | Pretrained baseline, no fine-tuning | 0 h | 0 h | — | 0 | Same baseline and **zero-shot Kidaw’ida** evaluation for Kinya-Flex |
| **E1** | MMS-TTS-Kin | Kinyarwanda-only fine-tuning | **17.89 h** | 0 h | — | **30** | How much can MMS improve its existing Kinyarwanda TTS capability? |
| **E2** | Kinya-Flex | Kinyarwanda-only fine-tuning | **17.89 h** | 0 h | — | **30** | How much can Kinya-Flex improve its existing Kinyarwanda TTS capability? |
| **E3** | MMS-TTS-Kin | Combined Kinyarwanda + Kidaw’ida | **17.89 h** | **~17.89 h subset** | **Balanced** | **50** | How well does MMS learn Kidaw’ida when both languages contribute approximately equal training volume? |
| **E4** | Kinya-Flex | Combined Kinyarwanda + Kidaw’ida | **17.89 h** | **~17.89 h subset** | **Balanced** | **50** | Same balanced bilingual experiment for Kinya-Flex |
| **E5** | MMS-TTS-Kin | Combined Kinyarwanda + full Kidaw’ida | **17.89 h** | **27.07 h** | **Unbalanced** | **50** | What happens when all available Kidaw’ida data are used instead of matching Kinyarwanda volume? |
| **E6** | Kinya-Flex | Combined Kinyarwanda + full Kidaw’ida | **17.89 h** | **27.07 h** | **Unbalanced** | **50** | Same full-data bilingual experiment for Kinya-Flex |

**Validation / early stop:** Kin-only (E1/E2) → combined Kin `dev.csv`. Bilingual (E3–E6) → Kin dev + Kidaw’ida dev (fixed rule).  
**Final eval (every condition):** FLEURS Kin (re-score after FT; B0/B1 already done); Kin tests separately (`test_mbaza`, `test_afrispeak_female`, `test_afrispeak_male`), then optional combined Kin test; Kidaw’ida test for zero-shot (B0/B1) and after bilingual FT (E3–E6).

### Metrics (every final test condition)

| Metric | Direction | What it measures |
|--------|-----------|------------------|
| **WER** | ↓ | Intelligibility at word level through ASR loopback |
| **CER** | ↓ | Intelligibility at character level |
| **UTMOS** | ↑ | Estimated speech naturalness |
| **Inference latency** | ↓ | Time required to generate speech |
| **RTF** | ↓ | Synthesis speed relative to audio duration |

Same ASR (Drive CTC), UTMOS, and hardware notes for all rows so B0/B1 ↔ E* are comparable. Details: `configs/metrics.yaml`.

### B0 / B1 — zero-shot Kidaw’ida (ASR expanded test)

Uses Kidaw’ida ASR **test** (~1004 clips / ~1.35 h). Kin models synthesize Dav text with **no Dav fine-tuning**.

```bash
conda activate healthtts
cd ~/HealthTTS

# 1) FLEURS-style manifests from ASR processed TSVs (once)
python scripts/prepare_kidawida_eval_manifest.py
# → …/kidawida_asr_drive/processed/kidawida/eval/{test,dev}.tsv

# 2) Synthesize (GPU) — smoke then full
LIMIT=20 bash scripts/submit_eval_mms_kidawida.sh
bash scripts/submit_eval_mms_kidawida.sh

LIMIT=20 bash scripts/submit_eval_flex_kidawida.sh
bash scripts/submit_eval_flex_kidawida.sh

# 3) Score (WER/CER/UTMOS/RTF) — same scorer as FLEURS
# MMS:
SYNTH_DIR=$BASELINES/eval_kidawida/mms_tts_kin/test \
  SLURM_JOB_NAME=score-mms-dav RETIME=0 \
  bash scripts/submit_score_tts_fleurs.sh
# Flex:
SYNTH_DIR=$BASELINES/eval_kidawida/kinya_flex_tts/test \
  SLURM_JOB_NAME=score-flex-dav RETIME=0 \
  bash scripts/submit_score_tts_fleurs.sh
```

Outputs under `$BASELINES/eval_kidawida/{mms_tts_kin,kinya_flex_tts}/test/`.

## Environment

```bash
conda create -n healthtts python=3.11 -y
conda activate healthtts
pip install -U "huggingface_hub[cli]" datasets
hf auth login
```

## Download datasets

### 1) Kinyarwanda TTS (Hugging Face)

```bash
conda activate healthtts
export DATA=/project/community/rmwisene/datasets

hf download mbazaNLP/kinyarwanda-tts-dataset \
  --repo-type dataset \
  --local-dir "$DATA/kinyarwanda_tts"
```

### 2) AfriSpeak Kinyarwanda Female + Male (Hugging Face)

Male repo is **gated** — open the HF page, accept terms, then `hf auth login`.

```bash
export DATA=/project/community/rmwisene/datasets
export HF_HOME=/project/community/rmwisene/.cache/huggingface
mkdir -p "$DATA"

hf download DigitalUmuganda/afrispeak_kinyarwanda_female_tts_dataset \
  --repo-type dataset \
  --local-dir "$DATA/afrispeak_kinyarwanda_female"

hf download DigitalUmuganda/afrispeak_kinyarwanda_male_tts_dataset \
  --repo-type dataset \
  --local-dir "$DATA/afrispeak_kinyarwanda_male"
```

Expected ~**2.61 h** female + ~**11.47 h** male (+ mbaza ~**6 h** → ~**20 h** total).

### 3) Merge Kin training pool (mbaza + AfriSpeak F/M)

16 kHz mono WAV; **no lowercasing**. Split policy:

- **AfriSpeak:** official `train` + `val`→dev (unchanged). Official test WAVs **not published** → hold out **~5%** of usable train per gender for test (`seed=42`; ~86 ♀ / ~440 ♂)
- **mbaza:** create reproducible train/dev/test (`seed=42`, 90/5/5)
- **Combined:** train/dev/test = union of all three sources

```bash
conda activate healthtts
cd ~/HealthTTS
pip install -q soundfile pyyaml librosa
python scripts/prepare_kinyarwanda_tts_corpus.py
# smoke: python scripts/prepare_kinyarwanda_tts_corpus.py --limit 50
```

Outputs under `/project/community/rmwisene/tts_data_cleaning/kinyarwanda_tts_mbaza_afrispeak/`:

```text
manifests/
  train.csv dev.csv test.csv          # combined
  test_mbaza.csv
  test_afrispeak_female.csv
  test_afrispeak_male.csv
  test_kin_combined.csv               # same as test.csv
  all.csv
```

Protocol: train → tune on **dev** → best ckpt → score each `test_*.csv` → report combined Kin test (+ **FLEURS** external).  
QC keeps repeated transcripts; only identical audio bytes are dropped.

### 4) Kidaw’ida — Mozilla Data Collective (not Hugging Face)

```bash
export MDC_API_KEY='YOUR_API_KEY'   # never commit
export DATA=/project/community/rmwisene/datasets
mkdir -p "$DATA/kidawida_cv27"
cd "$DATA/kidawida_cv27"

RESPONSE=$(curl -sS -X POST \
  "https://mozilladatacollective.com/api/datasets/cmu5w2bh700aso107ffpduqpp/download" \
  -H "Authorization: Bearer ${MDC_API_KEY}" \
  -H "Content-Type: application/json")

DOWNLOAD_URL=$(echo "$RESPONSE" | jq -r '.downloadUrl')
curl -L --fail -o common-voice-scripted-speech-27-0-kidaw-69655f89.tar.gz "$DOWNLOAD_URL"
tar -xzf common-voice-scripted-speech-27-0-kidaw-69655f89.tar.gz
# optional: rm -f common-voice-scripted-speech-27-0-kidaw-69655f89.tar.gz
```

Or: `MDC_API_KEY=... bash scripts/download_kidawida_mdc.sh`

### 4b) Prepare Dav training pool (CV train/dev/test)

Requires frozen cleaned transcripts first (audit + freeze, if not already done):

```bash
conda activate healthtts
cd ~/HealthTTS
# once:
# python scripts/audit_clean_kidawida_transcripts.py
# python scripts/freeze_cleaned_kidawida.py

pip install -q soundfile pyyaml librosa   # librosa helps with CV mp3
python scripts/prepare_kidawida_tts_corpus.py
# smoke: python scripts/prepare_kidawida_tts_corpus.py --limit 50
```

Outputs under `/project/community/rmwisene/tts_data_cleaning/kidawida_tts_prepared/`  
(`manifests/train.csv|dev.csv|test.csv`, 16 kHz WAV under `audio/`, stats, rejected).

Keeps official CV splits and repeated transcripts; only identical audio bytes are dropped. Flagged q/x/ä/~ rows stay in unless you change the config.

### 5) FLEURS Kinyarwanda — eval only (Hugging Face)

```bash
hf download mbazaNLP/fleurs-kinyarwanda \
  --repo-type dataset \
  --local-dir "$DATA/fleurs_kinyarwanda"
```

### Check

```bash
du -sh "$DATA"/kinyarwanda_tts "$DATA"/kidawida_cv27 "$DATA"/fleurs_kinyarwanda
ls "$DATA"
```

## Notes

- Kidaw’ida is from **Mozilla Data Collective**, not Hugging Face.
- **`fleurs_kinyarwanda` = baseline evaluation only** — never use it for training/adaptation.
- Never store large audio under `~/` (home quota); never commit API keys.

## Load baselines (Kin)

Checkpoints go on **project storage** (not in this git repo):

```text
/project/community/rmwisene/tts_baselines/
├── mms_tts_kin/          # facebook/mms-tts-kin
└── kinya_flex_tts/       # wixdivin/kinya-flex-tts  (kinya_flex_tts_base_trained.pt)
```

| Baseline | Hub | Role |
|----------|-----|------|
| MMS-TTS Kin | [facebook/mms-tts-kin](https://huggingface.co/facebook/mms-tts-kin) | Kin VITS baseline (Transformers) — eval on FLEURS |
| Kinya-Flex-TTS | [wixdivin/kinya-flex-tts](https://huggingface.co/wixdivin/kinya-flex-tts) | Kin Flex/VITS2 baseline (DeepKIN `.pt`) — eval on FLEURS |

### Download on Orchard

```bash
conda activate healthtts
cd ~/HealthTTS
pip install -r requirements.txt

export BASELINES=/project/community/rmwisene/tts_baselines
bash scripts/download_baselines.sh
```

### Kinya-Flex-TTS note

Weights download with the script (`kinya_flex_tts_base_trained.pt`). Runtime inference uses **DeepKIN** (`FlexKinyaTTS`). Speakers: `0` Female1, `1` Female2, `2` Male (24 kHz).

Dav has no public dedicated baseline yet; adaptation can init from MMS Kin + cleaned `kidawida_cv27`.

## Clean FLEURS for baseline evaluation

Eval data only (`test` + `dev`). **Train split is never written into eval manifests.**

```bash
conda activate healthtts
cd ~/HealthTTS

python scripts/clean_fleurs_eval.py --config configs/fleurs_eval.yaml
```

Outputs (on project):

```text
/project/community/rmwisene/datasets/fleurs_kinyarwanda/processed/eval/
├── test.tsv          # primary baseline eval
├── dev.tsv           # optional
└── summary.json      # kept / dropped counts
```

Cleaning rules: Unicode NFC, whitespace collapse, drop empty text, missing audio, and extreme durations (default 0.3–30s).

## Run MMS baseline on cleaned FLEURS (eval — not training)

FLEURS stays **evaluation only**. This synthesizes from cleaned `test.tsv` texts:

```bash
conda activate healthtts
export BASELINES=/project/community/rmwisene/tts_baselines
export HF_HOME=/project/community/rmwisene/.cache/huggingface

# prefer a GPU node for full test; login CPU ok for --limit 20
python scripts/eval_mms_fleurs.py --split test --limit 20
# full:
# python scripts/eval_mms_fleurs.py --split test
```

Outputs under `$BASELINES/eval_fleurs/mms_tts_kin/test/`.

**Adaptation / training** (later) uses `kinyarwanda_tts` + `kidawida_cv27`, not FLEURS.

## Agreed evaluation metrics

Canonical table: **[Experiments → Metrics](#metrics-every-final-test-condition)** above.  
Pipeline: Text → TTS → audio → **Drive CTC** → WER/CER; plus latency, RTF, UTMOS (`configs/metrics.yaml`).

ASR scorer: NeMo CTC from Drive  
https://drive.google.com/file/d/1gx4olc2XdlUNkNlFqXcOq1F4IbcVHyGI/view  
→ `/project/community/rmwisene/asr/combined-ctc-15-ep-nocl.nemo`

### Score existing MMS FLEURS synths (GPU)

```bash
cd ~/HealthTTS && git pull
conda activate healthtts
pip install jiwer

# 1) pull CTC from Drive with rclone (once)
bash scripts/download_ctc_asr.sh
# default src:
#   Gdrive-Okeyo:ASR Summer 2026/models/no-curriculum/combined-ctc-15-ep-nocl.nemo

# 2) score: WER/CER (CTC) + RTF + UTMOS
export BASELINES=/project/community/rmwisene/tts_baselines
export HF_HOME=/project/community/rmwisene/.cache/huggingface
bash scripts/submit_score_tts_fleurs.sh
```

Results:

```text
$BASELINES/eval_fleurs/mms_tts_kin/test/metrics.json
$BASELINES/eval_fleurs/mms_tts_kin/test/scores.tsv
```

## Kinya-Flex baseline on FLEURS

Weights: `$BASELINES/kinya_flex_tts/kinya_flex_tts_base_trained.pt`  
Code: DeepKIN-AgAI (install once).

```bash
# 1) install DeepKIN into healthtts (once)
bash scripts/setup_deepkin.sh

# 2) synthesize (GPU) — test then dev
SPLIT=test SPEAKER=0 bash scripts/submit_eval_flex_fleurs.sh
# or on an interactive GPU node:
# python scripts/eval_flex_fleurs.py --split test --speaker 0 --device cuda

SPLIT=dev SPEAKER=0 bash scripts/submit_eval_flex_fleurs.sh

# 3) score with Drive CTC + UTMOS (timing already in synthesis.tsv → no --retime needed)
SYNTH_DIR=$BASELINES/eval_fleurs/kinya_flex_tts/test \
  bash scripts/submit_score_tts_fleurs.sh
```

Speakers: `0` Female1, `1` Female2, `2` Male (24 kHz). Default eval speaker = `0`.
