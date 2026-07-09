#!/usr/bin/env bash
# Generate hidden states using transformers (4 workers, 2 GPU each).
# Single script, fire and forget. All 4 workers run in background, script waits.
#
# Usage:
#   EAGLE3=$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/eagle3/20260615 \
#     bash examples/train/maiprofile/gen_hidden_states_all.sh

set -euo pipefail

EAGLE3="${EAGLE3:?Set EAGLE3=\$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/eagle3/20260615}"
MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
PREPARED="$EAGLE3/prepared"
OUTPUT="$EAGLE3/hidden_states"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/gen_hidden_states_transformers.py"

mkdir -p "$OUTPUT"

# Resume: don't clean existing files. The python script skips existing hs_*.safetensors.
# To force a full re-run, manually: rm -f "$OUTPUT"/hs_*.safetensors

echo "============================================="
echo " Hidden States Generation (4 × tp=2 workers)"
echo "============================================="
echo "  model    : $MODEL"
echo "  prepared : $PREPARED"
echo "  output   : $OUTPUT"
echo ""

GPU_PAIRS=("0,1" "2,3" "4,5" "6,7")
PIDS=()

for rank in 0 1 2 3; do
    GPUS="${GPU_PAIRS[$rank]}"
    LOG="/tmp/gen_hs_worker_${rank}.log"
    echo "  Starting worker $rank (GPU=$GPUS, log=$LOG)"
    CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT" \
        --model "$MODEL" \
        --prepared-data "$PREPARED" \
        --output "$OUTPUT" \
        --world-size 4 \
        --rank "$rank" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "  All 4 workers launched. Logs: /tmp/gen_hs_worker_{0,1,2,3}.log"
echo "  Waiting for completion..."
echo ""

FAILED=0
for rank in 0 1 2 3; do
    if wait "${PIDS[$rank]}"; then
        echo "  Worker $rank finished OK"
    else
        echo "  Worker $rank FAILED (check /tmp/gen_hs_worker_${rank}.log)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
TOTAL_FILES=$(find "$OUTPUT" -name "hs_*.safetensors" | wc -l)
echo "============================================="
if [[ $FAILED -eq 0 ]]; then
    echo " ALL DONE. Files: $TOTAL_FILES"
else
    echo " $FAILED worker(s) failed. Files so far: $TOTAL_FILES"
    echo " Check /tmp/gen_hs_worker_*.log"
fi
echo " Output: $OUTPUT"
echo "============================================="
echo ""
echo "Next: train with"
echo "  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node 8 \\"
echo "    scripts/train.py \\"
echo "    --verifier-name-or-path $MODEL \\"
echo "    --data-path $PREPARED \\"
echo "    --hidden-states-path $OUTPUT \\"
echo "    --save-path $EAGLE3/checkpoints \\"
echo "    --draft-vocab-size 32000 --num-layers 1 --ttt-steps 3 \\"
echo "    --epochs 20 --lr 1e-4 --total-seq-len 8192 \\"
echo "    --on-missing skip --logger tensorboard --log-dir $EAGLE3/logs \\"
echo "    --run-name eagle3_maiprofile_26b"
