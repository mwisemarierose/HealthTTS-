#!/usr/bin/env bash
# Check synth dirs needed for UTMOS / naturalness listening pairs.
set -euo pipefail
dirs=(
  "/project/community/rmwisene/tts_baselines/eval_kidawida/mms_tts_kin/test"
  "/project/community/rmwisene/tts_baselines/eval_kidawida/mms_e3/test"
  "/project/community/rmwisene/tts_baselines/eval_kidawida/kinya_flex_tts/test"
  "/project/community/rmwisene/tts_baselines/eval_kidawida/flex_e4/test"
  "/project/community/rmwisene/tts_baselines/eval_kin_dev/flex_e2/dev"
  "/project/community/rmwisene/tts_baselines/eval_kin/flex_e4/dev"
  "/project/community/rmwisene/tts_baselines/eval_kin_dev/mms_e1/dev"
  "/project/community/rmwisene/tts_baselines/eval_kin/mms_e3/dev"
)
echo "Listening synth inventory:"
for d in "${dirs[@]}"; do
  n_wav=$(find "$d" -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
  has_meta="no"
  [[ -f "$d/synthesis.tsv" || -f "$d/scores.tsv" ]] && has_meta="yes"
  if [[ -d "$d" && "$n_wav" -gt 0 ]]; then
    printf "OK   %-70s wavs=%-5s meta=%s\n" "$d" "$n_wav" "$has_meta"
  else
    printf "MISS %-70s wavs=%-5s meta=%s\n" "$d" "$n_wav" "$has_meta"
  fi
done
