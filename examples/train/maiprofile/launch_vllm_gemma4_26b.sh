#!/usr/bin/env bash
# Step 2 — Launch vLLM to serve the Gemma4-26B-A4B verifier and extract hidden
# states for EAGLE-3 online training.
#
# Wraps speculators' scripts/launch_vllm.py, which configures vLLM with
# method=extract_hidden_states + a KV connector that writes hidden states to a
# shared path the trainer reads from.
#
# IMPORTANT — target-layer-ids:
#   If TARGET_LAYER_IDS is left empty, launch_vllm.py auto-picks
#   [2, N//2, N-3, N] (N = num_hidden_layers). Whatever it uses, the SAME ids
#   MUST be passed to train.py (Step 3). To stay unambiguous we let it default
#   here AND pass the same default explicitly in train (see README). If you pin
#   custom ids, set TARGET_LAYER_IDS here and mirror it in train_eagle3_maiprofile.sh.
#   Run probe_gemma4_26b.py first to see the real N and default ids for 26B-A4B.
#
# Run ON THE SERVER, inside the vLLM venv. Keep this running; start Step 3 in a
# second terminal on the same node (hidden states are shared via HIDDEN_STATES_PATH).
#
# Usage:
#   bash examples/train/maiprofile/launch_vllm_gemma4_26b.sh
#   # dry run to just print the vLLM command:
#   DRY_RUN=1 bash examples/train/maiprofile/launch_vllm_gemma4_26b.sh

set -euo pipefail

# ============ Configuration (override via env) ============
MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
VLLM_PORT="${VLLM_PORT:-8000}"

# GPUs for the verifier/datagen half of online training. On 8xA100, a common
# split is 4 GPUs for vLLM datagen + 4 for training; tune on the server.
VLLM_GPUS="${VLLM_GPUS:-0,1,2,3}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-4}"

# Shared location for extracted hidden states. MUST be reachable from the
# trainer process (same node, or shared network drive).
HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:-/tmp/hidden_states_maiprofile_26b}"

# Optional explicit target layer ids (space-separated). Empty => script default
# [2, N//2, N-3, N]. If you set this, mirror it EXACTLY in train (Step 3).
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-}"

# Extra args passed straight to vLLM after `--`.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
# ==========================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

echo "=== Step 2: Launch vLLM verifier (hidden-state extraction) ==="
echo "  model              : ${MODEL}"
echo "  vllm gpus          : ${VLLM_GPUS} (data-parallel-size=${DATA_PARALLEL_SIZE})"
echo "  port               : ${VLLM_PORT}"
echo "  hidden states path : ${HIDDEN_STATES_PATH}"
echo "  target-layer-ids   : ${TARGET_LAYER_IDS:-<auto: 2 N//2 N-3 N>}"

LAUNCH_ARGS=(python scripts/launch_vllm.py "${MODEL}"
    --hidden-states-path "${HIDDEN_STATES_PATH}")

if [[ -n "${TARGET_LAYER_IDS}" ]]; then
    # shellcheck disable=SC2206
    IDS_ARR=(${TARGET_LAYER_IDS})
    LAUNCH_ARGS+=(--target-layer-ids "${IDS_ARR[@]}")
fi

if [[ -n "${DRY_RUN:-}" ]]; then
    LAUNCH_ARGS+=(--dry-run)
fi

# Everything after `--` goes to vLLM.
VLLM_PASSTHROUGH=(-- --data-parallel-size "${DATA_PARALLEL_SIZE}"
    --port "${VLLM_PORT}"
    --gpu-memory-utilization "${GPU_MEM_UTIL}")
if [[ -n "${MAX_MODEL_LEN}" ]]; then
    VLLM_PASSTHROUGH+=(--max-model-len "${MAX_MODEL_LEN}")
fi

echo "+ CUDA_VISIBLE_DEVICES=${VLLM_GPUS} ${LAUNCH_ARGS[*]} ${VLLM_PASSTHROUGH[*]}"
CUDA_VISIBLE_DEVICES="${VLLM_GPUS}" "${LAUNCH_ARGS[@]}" "${VLLM_PASSTHROUGH[@]}"
