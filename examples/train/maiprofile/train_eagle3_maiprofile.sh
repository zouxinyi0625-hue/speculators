#!/usr/bin/env bash
# Step 3 — Train the EAGLE-3 draft for Gemma4-26B-A4B on MAI Profile data.
#
# Wraps speculators' scripts/train.py in online mode: hidden states are pulled
# on-demand from the live vLLM verifier launched in Step 2.
#
# Parameters here follow speculators' OFFICIAL DEFAULTS (goal: get the pipeline
# running first, tune later). The only thing we pin explicitly is
# --target-layer-ids, which MUST match what Step 2 used.
#
# Run ON THE SERVER, inside the speculators venv, in a SECOND terminal while
# Step 2's vLLM server is up. Needs GPUs distinct from the vLLM ones.
#
# Usage:
#   bash examples/train/maiprofile/train_eagle3_maiprofile.sh

set -euo pipefail

# ============ Configuration (override via env) ============
MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
DATA_PATH="${DATA_PATH:-./output/maiprofile_eagle3_26b}"     # from Step 1
VLLM_PORT="${VLLM_PORT:-8000}"
SAVE_PATH="${SAVE_PATH:-${DATA_PATH}/checkpoints}"

# --- Training GPUs (the OTHER half of the 8xA100 box vs Step 2's vLLM GPUs) ---
TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-4}"

# --- EAGLE-3 hyperparameters: speculators OFFICIAL DEFAULTS ---
#   DRAFT_VOCAB_SIZE 32000 : prune the 262k target vocab to a 32k draft vocab
#                            (needs token_freq.pt from Step 1). See README.
#   NUM_LAYERS       1     : draft = single llama-style decoder layer (default).
#   TTT_STEPS        3     : training-time-test unroll depth (default).
#   TTT_DECAY        1.0   : per-step loss weight decay (1.0 = equal weight).
#   EPOCHS           20    : train.py default.
#   LR               1e-4  : train.py default.
#   SEQ_LEN          8192  : must be >= the seq length used in Step 1.
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
NUM_LAYERS="${NUM_LAYERS:-1}"
TTT_STEPS="${TTT_STEPS:-3}"
TTT_DECAY="${TTT_DECAY:-1.0}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-1e-4}"
SEQ_LEN="${SEQ_LEN:-8192}"

# target-layer-ids: MUST equal Step 2. Empty => train.py default [2,N//2,N-3,N],
# which also matches launch_vllm.py's default. Pin explicitly only if you pinned
# it in Step 2. Run probe_gemma4_26b.py to get the real ids for 26B-A4B.
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-}"

# Optional experiment logger: trackio | wandb | tensorboard | mlflow
LOGGER="${LOGGER:-tensorboard}"
RUN_NAME="${RUN_NAME:-eagle3_maiprofile_26b_a4b}"
# ==========================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

echo "=== Step 3: Train EAGLE-3 draft on MAI Profile (26B-A4B) ==="
echo "  verifier         : ${MODEL}"
echo "  data path        : ${DATA_PATH}"
echo "  save path        : ${SAVE_PATH}"
echo "  train gpus       : ${TRAIN_GPUS} (nproc_per_node=${NUM_TRAIN_GPUS})"
echo "  draft vocab size : ${DRAFT_VOCAB_SIZE}"
echo "  num layers       : ${NUM_LAYERS}"
echo "  ttt steps        : ${TTT_STEPS} (decay ${TTT_DECAY})"
echo "  epochs / lr      : ${EPOCHS} / ${LR}"
echo "  total seq len    : ${SEQ_LEN}"
echo "  target-layer-ids : ${TARGET_LAYER_IDS:-<auto: 2 N//2 N-3 N>}"

TRAIN_ARGS=(scripts/train.py
    --verifier-name-or-path "${MODEL}"
    --data-path "${DATA_PATH}"
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
    --logger "${LOGGER}"
    --run-name "${RUN_NAME}")

if [[ -n "${TARGET_LAYER_IDS}" ]]; then
    # shellcheck disable=SC2206
    IDS_ARR=(${TARGET_LAYER_IDS})
    TRAIN_ARGS+=(--target-layer-ids "${IDS_ARR[@]}")
fi

echo "+ CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} torchrun --standalone --nproc_per_node ${NUM_TRAIN_GPUS} ${TRAIN_ARGS[*]}"
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" torchrun \
    --standalone --nproc_per_node "${NUM_TRAIN_GPUS}" \
    "${TRAIN_ARGS[@]}"

echo "=== Done. Checkpoints under: ${SAVE_PATH} ==="
echo "    checkpoint_best -> lowest val-loss epoch; deploy that in vLLM."
