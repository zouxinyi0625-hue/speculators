#!/usr/bin/env bash
# Step 0 — Regenerate assistant responses for ALL maiprofile layers using the
# 26B-A4B target model (via vLLM chat completions).
#
# This is required BEFORE prepare_data (Step 1) because EAGLE-3's loss mask
# only covers assistant-response spans — prompt-only samples yield zero loss.
#
# Uses speculators' response_regeneration pattern: start vLLM → hit the chat
# completions endpoint → save conversations jsonl with the assistant turn.
# The 26B-A4B model (our actual target) generates the responses — critical for
# distribution alignment (DSpark's 12B regen is NOT usable here).
#
# Output: a single merged JSONL with all layers' conversations (system + user +
# assistant), ready for prepare_data.py in Step 1.
#
# Run ON THE SERVER, inside the vLLM venv (needs GPU + model weights).
#
# Usage:
#   bash examples/train/maiprofile/regenerate_maiprofile.sh
#
# To regenerate only some layers (raw mode):
#   LAYERS="layer1_actual,layer3_seasonality" bash .../regenerate_maiprofile.sh
#
# To preserve DSpark's train/eval split (recommended for comparability):
#   INPUT_FILE=$AZURE_ML_INPUT_msndni/.../train_maiprofile_all_layers.jsonl \
#     bash examples/train/maiprofile/regenerate_maiprofile.sh
#   This reads the pre-split train jsonl, strips old (12B) assistant turns,
#   and regenerates them with the 26B-A4B target. Eval stays untouched.

set -euo pipefail

# ============ Configuration (override via env) ============
MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
PORT="${PORT:-8000}"

# GPUs for vLLM (all 8 for fastest regen; training doesn't run during this step)
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
DP_SIZE="${DP_SIZE:-4}"
TP_SIZE="${TP_SIZE:-2}"

# Data paths — all under the eagle3/ tree on mount
MSNDNI="${AZURE_ML_INPUT_msndni:?AZURE_ML_INPUT_msndni is not set (Azure ML mount)}"
DATE="${DATE:-20260615}"
RAW_DIR="${RAW_DIR:-${MSNDNI}/shares/users/zxy/maiprofile/raw_data/${DATE}}"
EAGLE3_DIR="${EAGLE3_DIR:-${MSNDNI}/shares/users/zxy/maiprofile/eagle3/${DATE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${EAGLE3_DIR}/regen_26b}"
OUTFILE="${OUTFILE:-${OUTPUT_DIR}/train_all_layers_regen_26b.jsonl}"

# Input: either INPUT_FILE (pre-split train jsonl, recommended) or raw layers.
# Default = the split produced by split_maiprofile_eagle3.py in the eagle3 dir.
INPUT_FILE="${INPUT_FILE:-${EAGLE3_DIR}/train_all_layers.jsonl}"

# ALL layers by default (only used when INPUT_FILE is empty/missing)
LAYERS="${LAYERS:-layer1_actual,layer1_delta,layer1_intent,layer2_coarse_interest,layer2_temporal,layer3_commercial_interests,layer3_persona,layer3_seasonality,layer4_biography,layer4_commercial_preference}"

# Generation parameters
MAX_TOKENS="${MAX_TOKENS:-4096}"
CONCURRENCY="${CONCURRENCY:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
# ==========================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==========================================="
echo " Step 0: Regenerate MAI Profile responses"
echo "==========================================="
echo "  model         : ${MODEL}"
echo "  raw dir       : ${RAW_DIR}"
echo "  output file   : ${OUTFILE}"
echo "  layers        : ${LAYERS}"
echo "  max tokens    : ${MAX_TOKENS}"
echo "  concurrency   : ${CONCURRENCY}"
echo "  gpus          : ${GPUS} (dp=${DP_SIZE}, tp=${TP_SIZE})"
echo ""

# --- Sub-step A: Start vLLM ---
echo "--- Starting vLLM server on port ${PORT} ---"
VLLM_CMD=(vllm serve "${MODEL}" --host 127.0.0.1 --port "${PORT}" --api-key ""
    --data-parallel-size "${DP_SIZE}" --tensor-parallel-size "${TP_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}")
echo "  ${VLLM_CMD[*]}"
CUDA_VISIBLE_DEVICES="${GPUS}" "${VLLM_CMD[@]}" > "${SCRIPT_DIR}/regen_vllm.log" 2>&1 &
VLLM_PID=$!

cleanup() {
    if kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "Stopping vLLM server (PID ${VLLM_PID})..."
        kill "${VLLM_PID}"; sleep 3
        kill -0 "${VLLM_PID}" 2>/dev/null && kill -9 "${VLLM_PID}" || true
    fi
}
trap cleanup EXIT

echo "  Waiting for server (26B-A4B may take several minutes)..."
ENDPOINT="http://127.0.0.1:${PORT}/v1/models"
for i in $(seq 1 600); do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "  [FATAL] vLLM died. Last 20 lines:"; tail -20 "${SCRIPT_DIR}/regen_vllm.log"; exit 1
    fi
    curl -sf --connect-timeout 5 --max-time 10 "${ENDPOINT}" >/dev/null 2>&1 && break
    [ $((i % 15)) -eq 0 ] && echo "  Still loading... (${i}s)"
    sleep 2
done
echo "  Server ready."
echo ""

# --- Sub-step B: Run regeneration ---
echo "--- Running regeneration ---"
mkdir -p "$(dirname "${OUTFILE}")"

REGEN_ARGS=(python "${SCRIPT_DIR}/regenerate_maiprofile.py"
    --outfile "${OUTFILE}"
    --endpoint "http://127.0.0.1:${PORT}/v1/chat/completions"
    --model "${MODEL}"
    --max-tokens "${MAX_TOKENS}"
    --concurrency "${CONCURRENCY}"
    --resume)

if [[ -n "${INPUT_FILE:-}" && -f "${INPUT_FILE}" ]]; then
    echo "  mode: --input-file (preserving DSpark-style split)"
    echo "  file: ${INPUT_FILE}"
    REGEN_ARGS+=(--input-file "${INPUT_FILE}")
else
    if [[ -n "${INPUT_FILE:-}" && ! -f "${INPUT_FILE}" ]]; then
        echo "  [INFO] INPUT_FILE not found: ${INPUT_FILE}"
        echo "         Falling back to --raw-dir mode. Run split_maiprofile_eagle3.py first"
        echo "         if you want to preserve a train/eval split."
    fi
    echo "  mode: --raw-dir + --layers"
    echo "  dir : ${RAW_DIR}"
    echo "  layers: ${LAYERS}"
    REGEN_ARGS+=(--raw-dir "${RAW_DIR}" --layers "${LAYERS}")
fi

echo "+ ${REGEN_ARGS[*]}"
"${REGEN_ARGS[@]}"

echo ""
echo "==========================================="
echo " Done. Output: ${OUTFILE}"
echo "==========================================="
