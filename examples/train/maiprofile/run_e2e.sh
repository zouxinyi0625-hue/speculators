#!/usr/bin/env bash
# End-to-end EAGLE-3 maiprofile training: prepare → vLLM → train → cleanup.
# Single script, fire and forget. vLLM runs in background, train blocks until
# done, vLLM is killed on exit (or error).
#
# Prerequisites:
#   - Regen already done (train_all_layers_regen_26b.jsonl exists)
#   - speculators installed in current venv (pip install -e .)
#   - vLLM installed in current venv (or same venv with both)
#
# Usage:
#   bash examples/train/maiprofile/run_e2e.sh
#
# For pilot (quick validation):
#   EPOCHS=3 bash examples/train/maiprofile/run_e2e.sh

set -euo pipefail

# ============ Configuration ============
MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
VLLM_PORT="${VLLM_PORT:-8000}"

# Data paths
MSNDNI="${AZURE_ML_INPUT_msndni:?AZURE_ML_INPUT_msndni is not set}"
DATE="${DATE:-20260615}"
EAGLE3_DIR="${EAGLE3_DIR:-${MSNDNI}/shares/users/zxy/maiprofile/eagle3/${DATE}}"
REGEN_FILE="${REGEN_FILE:-${EAGLE3_DIR}/regen_26b/train_all_layers_regen_26b.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/maiprofile_eagle3_26b}"
SAVE_PATH="${SAVE_PATH:-${OUTPUT_DIR}/checkpoints}"

# GPU split: vLLM (hidden state extraction) vs training
VLLM_GPUS="${VLLM_GPUS:-0,1,2,3}"
VLLM_DP_SIZE="${VLLM_DP_SIZE:-4}"
TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-4}"

# Training hyperparams (speculators defaults, override via env)
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
NUM_LAYERS="${NUM_LAYERS:-1}"
TTT_STEPS="${TTT_STEPS:-3}"
TTT_DECAY="${TTT_DECAY:-1.0}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-1e-4}"
SEQ_LEN="${SEQ_LEN:-8192}"
MIN_VALID_TOKENS="${MIN_VALID_TOKENS:-14}"

# Hidden states shared path
HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:-/tmp/hidden_states_maiprofile_26b}"

# vLLM settings
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-900}"  # 15 min for 26B load
# =======================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "$(dirname "${SAVE_PATH}")"

echo "============================================="
echo " EAGLE-3 MAI Profile E2E Training Pipeline"
echo "============================================="
echo "  model          : ${MODEL}"
echo "  regen file     : ${REGEN_FILE}"
echo "  output dir     : ${OUTPUT_DIR}"
echo "  save path      : ${SAVE_PATH}"
echo "  vllm gpus      : ${VLLM_GPUS} (dp=${VLLM_DP_SIZE})"
echo "  train gpus     : ${TRAIN_GPUS} (nproc=${NUM_TRAIN_GPUS})"
echo "  epochs/lr      : ${EPOCHS} / ${LR}"
echo "  seq len        : ${SEQ_LEN}"
echo "  draft vocab    : ${DRAFT_VOCAB_SIZE}"
echo "  ttt steps      : ${TTT_STEPS}"
echo ""

# Validate regen file exists
if [[ ! -f "${REGEN_FILE}" ]]; then
    echo "[FATAL] Regen file not found: ${REGEN_FILE}"
    echo "        Run regenerate_maiprofile.sh first."
    exit 1
fi

VLLM_PID=""
cleanup() {
    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo ""
        echo "--- Stopping vLLM server (PID ${VLLM_PID}) ---"
        kill "${VLLM_PID}" 2>/dev/null || true
        sleep 3
        kill -0 "${VLLM_PID}" 2>/dev/null && kill -9 "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ===== Step 1: Prepare data =====
echo ""
echo "===== Step 1/3: Prepare data ====="
PREP_CMD=(python scripts/prepare_data.py
    --model "${MODEL}"
    --data "${REGEN_FILE}"
    --output "${OUTPUT_DIR}"
    --seq-length "${SEQ_LEN}"
    --minimum-valid-tokens "${MIN_VALID_TOKENS}"
    --num-preprocessing-workers 8)
echo "+ ${PREP_CMD[*]}"
"${PREP_CMD[@]}" 2>&1 | tee "${LOG_DIR}/prepare_data.log"
echo "  Done. Dataset at: ${OUTPUT_DIR}"
echo ""

# ===== Step 2: Launch vLLM (background) =====
echo "===== Step 2/3: Launch vLLM (hidden state extraction) ====="
VLLM_CMD=(python scripts/launch_vllm.py "${MODEL}"
    --hidden-states-path "${HIDDEN_STATES_PATH}"
    -- --data-parallel-size "${VLLM_DP_SIZE}"
    --port "${VLLM_PORT}"
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
    --no-enable-chunked-prefill)
echo "+ CUDA_VISIBLE_DEVICES=${VLLM_GPUS} ${VLLM_CMD[*]}"
CUDA_VISIBLE_DEVICES="${VLLM_GPUS}" "${VLLM_CMD[@]}" > "${LOG_DIR}/vllm_server.log" 2>&1 &
VLLM_PID=$!
echo "  vLLM PID: ${VLLM_PID}"
echo "  Log: ${LOG_DIR}/vllm_server.log"

# Wait for server ready
echo "  Waiting for vLLM (26B-A4B may take several minutes)..."
ENDPOINT="http://127.0.0.1:${VLLM_PORT}/health"
ELAPSED=0
while true; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "  [FATAL] vLLM died. Last 30 lines:"
        tail -30 "${LOG_DIR}/vllm_server.log"
        exit 1
    fi
    if curl -sf --connect-timeout 5 --max-time 10 "${ENDPOINT}" >/dev/null 2>&1; then
        echo "  Server ready (${ELAPSED}s)."
        break
    fi
    ELAPSED=$((ELAPSED + 2))
    if [[ ${ELAPSED} -ge ${VLLM_READY_TIMEOUT} ]]; then
        echo "  [FATAL] vLLM not ready after ${VLLM_READY_TIMEOUT}s. Last 30 lines:"
        tail -30 "${LOG_DIR}/vllm_server.log"
        exit 1
    fi
    [[ $((ELAPSED % 30)) -eq 0 ]] && echo "  Still loading... (${ELAPSED}s)"
    sleep 2
done
echo ""

# ===== Step 3: Train =====
echo "===== Step 3/3: Train EAGLE-3 draft ====="
TRAIN_ARGS=(scripts/train.py
    --verifier-name-or-path "${MODEL}"
    --data-path "${OUTPUT_DIR}"
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1"
    --save-path "${SAVE_PATH}"
    --draft-vocab-size "${DRAFT_VOCAB_SIZE}"
    --num-layers "${NUM_LAYERS}"
    --ttt-steps "${TTT_STEPS}"
    --ttt-step-loss-decay "${TTT_DECAY}"
    --epochs "${EPOCHS}"
    --lr "${LR}"
    --total-seq-len "${SEQ_LEN}"
    --on-missing generate
    --on-generate delete
    --logger tensorboard
    --log-dir "${LOG_DIR}"
    --run-name "eagle3_maiprofile_26b_a4b")

echo "+ CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} torchrun --standalone --nproc_per_node ${NUM_TRAIN_GPUS} ${TRAIN_ARGS[*]}"
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" torchrun \
    --standalone --nproc_per_node "${NUM_TRAIN_GPUS}" \
    "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/train.log"

echo ""
echo "============================================="
echo " E2E DONE"
echo "============================================="
echo "  Checkpoints: ${SAVE_PATH}/"
echo "  Best:        ${SAVE_PATH}/checkpoint_best"
echo "  Logs:        ${LOG_DIR}/"
echo "  vLLM log:    ${LOG_DIR}/vllm_server.log"
echo "  Train log:   ${LOG_DIR}/train.log"
echo ""
echo "  Next: deploy checkpoint_best in vLLM and bench against MTP baseline."
