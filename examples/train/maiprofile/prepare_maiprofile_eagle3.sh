#!/usr/bin/env bash
# Step 1 — Prepare MAI Profile data for EAGLE-3 training (Gemma4-26B-A4B).
#
# Wraps speculators' scripts/prepare_data.py. Reuses the SAME maiprofile
# short-layer prompt split that the DSpark line already produced, because that
# split is already in the `conversations` jsonl schema that prepare_data.py
# expects for custom data.
#
# EAGLE-3 online training only needs the PROMPTS here: assistant responses +
# hidden states are produced on-the-fly by the live vLLM verifier in Step 3.
# So we do NOT need DSpark's regenerated/ or target_cache/ artifacts.
#
# Output: preprocessed arrow dataset + token_freq.pt (for the 32k vocab map)
# under $OUTPUT_DIR. This step is the same for online and offline training.
#
# Run ON THE SERVER, inside the speculators venv.
#
# Usage:
#   bash examples/train/maiprofile/prepare_maiprofile_eagle3.sh
# Override any variable inline, e.g.:
#   OUTPUT_DIR=/data/eagle3_out MAX_SAMPLES=2000 bash .../prepare_maiprofile_eagle3.sh

set -euo pipefail

# ============ Configuration (override via env) ============
# Target/verifier model. Full 26B-A4B (MoE) instruct model.
MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"

# MAI Profile short-layer train split produced by the DSpark data pipeline.
# Same file the DSpark line trains on; it is already `conversations` jsonl.
MSNDNI="${AZURE_ML_INPUT_msndni:?AZURE_ML_INPUT_msndni is not set (Azure ML mount)}"
DATE="${DATE:-20260615}"
DATASET="${DATASET:-${MSNDNI}/shares/users/zxy/maiprofile/prepared_prompts/${DATE}/short_layers/train_maiprofile_short_layers.jsonl}"

# Where preprocessed data lands (arrow shards + token_freq.pt).
OUTPUT_DIR="${OUTPUT_DIR:-./output/maiprofile_eagle3_26b}"

# Preprocessing knobs. Defaults follow the official EAGLE-3 online tutorial;
# override to taste on the server.
#   MAX_SAMPLES: cap for a quick pipeline sanity run (empty = use all samples).
#   SEQ_LENGTH : max sequence length at preprocessing time.
MAX_SAMPLES="${MAX_SAMPLES:-}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# ==========================================================

echo "=== Step 1: Prepare MAI Profile data for EAGLE-3 ==="
echo "  target model : ${MODEL}"
echo "  dataset      : ${DATASET}"
echo "  output dir   : ${OUTPUT_DIR}"
echo "  seq length   : ${SEQ_LENGTH}"
echo "  max samples  : ${MAX_SAMPLES:-<all>}"

if [[ ! -f "${DATASET}" ]]; then
    echo "[FATAL] dataset not found: ${DATASET}" >&2
    echo "        Check the DSpark maiprofile prepare step ran and DATE is correct." >&2
    exit 1
fi

# Resolve to the repo root so scripts/ is importable regardless of CWD.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

CMD=(python scripts/prepare_data.py
    --model "${MODEL}"
    --data "${DATASET}"
    --output "${OUTPUT_DIR}"
    --seq-length "${SEQ_LENGTH}"
    --num-preprocessing-workers "${NUM_WORKERS}")

if [[ -n "${MAX_SAMPLES}" ]]; then
    CMD+=(--max-samples "${MAX_SAMPLES}")
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"

echo "=== Done. Preprocessed data at: ${OUTPUT_DIR} ==="
echo "    Expect: data-*.arrow, dataset_info.json, state.json, token_freq.pt"
