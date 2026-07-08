#!/usr/bin/env bash
# Step 1 — Prepare MAI Profile data for EAGLE-3 training (Gemma4-26B-A4B).
#
# Wraps speculators' scripts/prepare_data.py.
#
# IMPORTANT — this MUST consume the DSpark *regenerated* data (prompts WITH
# target-generated assistant responses), NOT the raw prompt split. EAGLE-3's
# loss mask is created only over assistant-response spans
# (preprocessing.py:_create_loss_mask_from_offsets); a system+user-only sample
# yields an all-zero loss mask, gets a "No assistant response spans" warning,
# and is dropped by --minimum-valid-tokens. The MAI Profile raw_data is
# prompt-only, so it CANNOT be fed here directly.
#
# DSpark already regenerated the short layers (36,103 rows, 0 errors) into a
# `conversations` jsonl that includes the assistant turn -> reuse it as-is.
# (online vLLM in Step 3 generates HIDDEN STATES, not the assistant text.)
#
# WARNING — valid-sample loss: MAI Profile prompts are long and assistant
# responses are comparatively short (DSpark regen used max_tokens=2048). If
# SEQ_LENGTH is too small, the assistant suffix gets truncated and too few
# supervised tokens remain, so many samples are dropped. Keep SEQ_LENGTH large
# (>=8192) so the assistant span survives. See DeepSpec maiprofile_data_overview.md.
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

# MAI Profile data. MUST include target-generated assistant responses.
# Use the DSpark REGENERATED file (not the raw prompt split).
# This file is already `conversations` jsonl with an assistant turn appended.
#
# DEFAULT: short-layer regen (36k rows). To use ALL layers (recommended —
# EAGLE-3 benefits from more data), set DATASET to point at a full-layer
# regenerated file, or concatenate multiple regen files into one jsonl.
# Long layers are NOT excluded — EAGLE-3 handles them fine at seq-length=8192.
MSNDNI="${AZURE_ML_INPUT_msndni:?AZURE_ML_INPUT_msndni is not set (Azure ML mount)}"
DATE="${DATE:-20260615}"
EAGLE3_DIR="${EAGLE3_DIR:-${MSNDNI}/shares/users/zxy/maiprofile/eagle3/${DATE}}"
DATASET="${DATASET:-${EAGLE3_DIR}/regen_26b/train_all_layers_regen_26b.jsonl}"

# Where preprocessed data lands (arrow shards + token_freq.pt).
OUTPUT_DIR="${OUTPUT_DIR:-./output/maiprofile_eagle3_26b}"

# Preprocessing knobs.
#   MAX_SAMPLES: cap for a quick pipeline sanity run (empty = use all samples).
#   SEQ_LENGTH : max sequence length. Keep LARGE (>=8192) so the (short)
#                assistant suffix survives truncation and the sample stays valid.
#   MIN_VALID_TOKENS: drop samples with fewer than this many trainable
#                (assistant) tokens after masking. Guards against near-empty
#                supervision from over-truncated long prompts.
MAX_SAMPLES="${MAX_SAMPLES:-}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
MIN_VALID_TOKENS="${MIN_VALID_TOKENS:-14}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# ==========================================================

echo "=== Step 1: Prepare MAI Profile data for EAGLE-3 ==="
echo "  target model : ${MODEL}"
echo "  dataset      : ${DATASET}"
echo "  output dir   : ${OUTPUT_DIR}"
echo "  seq length   : ${SEQ_LENGTH}"
echo "  min valid tok: ${MIN_VALID_TOKENS}"
echo "  max samples  : ${MAX_SAMPLES:-<all>}"

if [[ ! -f "${DATASET}" ]]; then
    echo "[FATAL] dataset not found: ${DATASET}" >&2
    echo "        This must be the DSpark REGENERATED file (with assistant turns)." >&2
    echo "        Check the DSpark regenerate step ran and DATE is correct." >&2
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
    --minimum-valid-tokens "${MIN_VALID_TOKENS}"
    --num-preprocessing-workers "${NUM_WORKERS}")

if [[ -n "${MAX_SAMPLES}" ]]; then
    CMD+=(--max-samples "${MAX_SAMPLES}")
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"

echo "=== Done. Preprocessed data at: ${OUTPUT_DIR} ==="
echo "    Expect: data-*.arrow, dataset_info.json, state.json, token_freq.pt"
