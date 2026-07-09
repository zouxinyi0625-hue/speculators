#!/usr/bin/env bash
# Offline hidden states generation: 4 independent vLLM instances (tp=2 each),
# each with LOW concurrency to avoid KV cache preemption causing partial writes.
#
# Root cause of "hidden states length mismatch": high concurrency causes vLLM to
# preempt requests mid-prefill, writing partial/empty hidden states files.
# Fix: concurrency=2 per instance, 4 instances for throughput.
#
# Usage:
#   EAGLE3=$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/eagle3/20260615 \
#     bash examples/train/maiprofile/gen_hidden_states.sh
#
# Resumes automatically (skips existing hs_*.safetensors files).

set -euo pipefail

EAGLE3="${EAGLE3:?Set EAGLE3=\$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/eagle3/20260615}"
MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
CONCURRENCY="${CONCURRENCY:-2}"  # LOW per instance — prevents preemption
NUM_INSTANCES=4
BASE_PORT=8000

PREPARED="$EAGLE3/prepared"
HS_OUTPUT="$EAGLE3/hidden_states"

mkdir -p "$HS_OUTPUT"

if [[ ! -d "$PREPARED" ]]; then
    echo "[FATAL] Prepared data not found: $PREPARED"
    echo "        Run prepare_data.py first."
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Cleanup function
PIDS=()
cleanup() {
    echo ""
    echo "--- Stopping all vLLM instances ---"
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 3
    for pid in "${PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done
    # Also kill any data_generation_offline workers
    pkill -f "data_generation_offline" 2>/dev/null || true
    echo "Cleanup done."
}
trap cleanup EXIT

echo "============================================="
echo " Offline Hidden States Generation (4 × tp=2)"
echo "============================================="
echo "  model       : $MODEL"
echo "  prepared    : $PREPARED"
echo "  output      : $HS_OUTPUT"
echo "  concurrency : $CONCURRENCY per instance"
echo "  instances   : $NUM_INSTANCES"
echo ""

# --- Start 4 vLLM instances ---
echo "--- Starting $NUM_INSTANCES vLLM instances ---"
GPU_PAIRS=("0,1" "2,3" "4,5" "6,7")
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i))
    GPUS="${GPU_PAIRS[$i]}"
    LOG="/tmp/vllm_hs_${i}.log"
    echo "  Instance $i: GPU=$GPUS port=$PORT log=$LOG"
    CUDA_VISIBLE_DEVICES="$GPUS" python scripts/launch_vllm.py "$MODEL" \
        --hidden-states-path "$HS_OUTPUT" \
        -- --tensor-parallel-size 2 --port "$PORT" \
        --gpu-memory-utilization 0.90 --max-model-len 16384 \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

# Wait for all instances to be ready
echo ""
echo "--- Waiting for all instances to be ready ---"
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i))
    ELAPSED=0
    while true; do
        if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
            echo "  [FATAL] Instance $i died. Log: /tmp/vllm_hs_${i}.log"
            tail -20 "/tmp/vllm_hs_${i}.log"
            exit 1
        fi
        if curl -sf --connect-timeout 5 --max-time 10 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            echo "  Instance $i ready (port $PORT, ${ELAPSED}s)"
            break
        fi
        ELAPSED=$((ELAPSED + 2))
        if [[ $ELAPSED -ge 900 ]]; then
            echo "  [FATAL] Instance $i timeout. Log: /tmp/vllm_hs_${i}.log"
            tail -20 "/tmp/vllm_hs_${i}.log"
            exit 1
        fi
        [[ $((ELAPSED % 60)) -eq 0 ]] && echo "  Instance $i still loading... (${ELAPSED}s)"
        sleep 2
    done
done
echo "  All instances ready."
echo ""

# --- Run 4 data generation workers ---
echo "--- Starting $NUM_INSTANCES data generation workers ---"
WORKER_PIDS=()
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i))
    echo "  Worker $i: endpoint=localhost:$PORT rank=$i/$NUM_INSTANCES concurrency=$CONCURRENCY"
    python scripts/data_generation_offline.py \
        --preprocessed-data "$PREPARED" \
        --endpoint "http://localhost:${PORT}/v1" \
        --output "$HS_OUTPUT" \
        --concurrency "$CONCURRENCY" \
        --world-size "$NUM_INSTANCES" \
        --rank "$i" \
        --validate-outputs \
        > "/tmp/datagen_${i}.log" 2>&1 &
    WORKER_PIDS+=($!)
done

echo ""
echo "--- Waiting for all workers to finish ---"
echo "  Logs: /tmp/datagen_{0,1,2,3}.log"
echo "  (This will take a few hours for 70k samples...)"
FAILED=0
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    if wait "${WORKER_PIDS[$i]}"; then
        echo "  Worker $i finished OK"
    else
        echo "  Worker $i FAILED (check /tmp/datagen_${i}.log)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================="
if [[ $FAILED -eq 0 ]]; then
    echo " ALL DONE. Hidden states at: $HS_OUTPUT"
    GENERATED=$(find "$HS_OUTPUT" -name "hs_*.safetensors" | wc -l)
    echo " Files generated: $GENERATED"
else
    echo " $FAILED worker(s) failed. Check logs."
fi
echo "============================================="
echo ""
echo "Next: kill this script (Ctrl+C or let cleanup run), then train:"
echo ""
echo "  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node 8 \\"
echo "    scripts/train.py \\"
echo "    --verifier-name-or-path $MODEL \\"
echo "    --data-path $PREPARED \\"
echo "    --hidden-states-path $HS_OUTPUT \\"
echo "    --save-path $EAGLE3/checkpoints \\"
echo "    --draft-vocab-size 32000 --num-layers 1 --ttt-steps 3 \\"
echo "    --epochs 20 --lr 1e-4 --total-seq-len 8192 \\"
echo "    --on-missing skip --logger tensorboard --log-dir $EAGLE3/logs \\"
echo "    --run-name eagle3_maiprofile_26b"
