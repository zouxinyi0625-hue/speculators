#!/usr/bin/env python3
"""Generate hidden states using vLLM's offline LLM API (no server needed).

Uses vLLM's LLM class directly with extract_hidden_states mode. This bypasses
all the server/HTTP/KV-connector issues we hit with launch_vllm.py + data_generation_offline.py.

Output: hs_<idx>.safetensors files compatible with speculators train.py.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python examples/train/maiprofile/gen_hidden_states_vllm_offline.py \
        --model google/gemma-4-26B-A4B-it \
        --prepared-data $EAGLE3/prepared \
        --output $EAGLE3/hidden_states_offline \
        --tp-size 2

Supports --world-size/--rank for multi-process parallelism.
Supports resume (skips existing hs_*.safetensors).
"""

import argparse
import gc
import tempfile
import os
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from safetensors.torch import save_file
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Target model (HF id or local path)")
    p.add_argument("--prepared-data", required=True, help="Path to prepared arrow dataset")
    p.add_argument("--output", required=True, help="Output dir for hs_*.safetensors")
    p.add_argument("--target-layer-ids", type=int, nargs="+", default=None,
                   help="Layer IDs to extract. Default: [2, N//2, N-3, N]")
    p.add_argument("--tp-size", type=int, default=2, help="Tensor parallel size (default: 2 for 26B)")
    p.add_argument("--max-model-len", type=int, default=8192, help="Max model length")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-samples", type=int, default=None, help="Limit samples (for testing)")
    p.add_argument("--world-size", type=int, default=1, help="Total parallel workers")
    p.add_argument("--rank", type=int, default=0, help="This worker's rank")
    p.add_argument("--batch-size", type=int, default=16, help="Requests per batch")
    return p.parse_args()


def get_target_layer_ids(model_name, explicit_ids=None):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(config, "text_config"):
        config = config.text_config
    N = config.num_hidden_layers
    if explicit_ids:
        return explicit_ids
    return [2, N // 2, N - 3, N]


def main():
    args = parse_args()

    from vllm import LLM, SamplingParams
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import example_hidden_states_connector

    # Determine target layers
    target_layer_ids = get_target_layer_ids(args.model, args.target_layer_ids)
    print(f"Model: {args.model}")
    print(f"Target layer IDs: {target_layer_ids}")
    print(f"TP size: {args.tp_size}")
    print()

    # Load dataset
    print(f"Loading prepared dataset from {args.prepared_data}...")
    dataset = load_from_disk(args.prepared_data)
    total = len(dataset)
    if args.max_samples:
        total = min(total, args.max_samples)

    # Compute this worker's indices
    all_indices = list(range(total))
    if args.world_size > 1:
        chunk_size = (total + args.world_size - 1) // args.world_size
        start = args.rank * chunk_size
        end = min(start + chunk_size, total)
        my_indices = all_indices[start:end]
        print(f"  Worker {args.rank}/{args.world_size}: indices [{start}, {end}) = {len(my_indices)} samples")
    else:
        my_indices = all_indices
    print(f"  Total: {total}, this worker: {len(my_indices)}")

    # Check existing (resume)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    existing = set()
    for f in output_path.glob("hs_*.safetensors"):
        try:
            idx = int(f.stem.split("_")[1])
            existing.add(idx)
        except (IndexError, ValueError):
            pass
    remaining = [idx for idx in my_indices if idx not in existing]
    if existing:
        print(f"  Found {len(existing & set(my_indices))} existing, {len(remaining)} remaining")
    if not remaining:
        print("  Nothing to do.")
        return
    print()

    # Create temp dir for vLLM's hidden states connector
    tmpdir = tempfile.mkdtemp(prefix="vllm_hs_")
    print(f"Temp dir for hidden states: {tmpdir}")

    # Initialize vLLM
    print("Initializing vLLM...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": target_layer_ids,
                },
            },
        },
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleHiddenStatesConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "shared_storage_path": tmpdir,
            },
        ),
    )
    sampling_params = SamplingParams(max_tokens=1)
    print("  vLLM ready.")
    print()

    # Process in batches
    print(f"Generating hidden states for {len(remaining)} samples (batch_size={args.batch_size})...")
    errors = 0
    generated = 0

    for batch_start in tqdm(range(0, len(remaining), args.batch_size),
                            desc=f"Worker {args.rank}",
                            total=(len(remaining) + args.batch_size - 1) // args.batch_size):
        batch_indices = remaining[batch_start:batch_start + args.batch_size]
        batch_prompts = []
        batch_token_ids = []

        for idx in batch_indices:
            sample = dataset[idx]
            input_ids = sample["input_ids"]
            if isinstance(input_ids, list):
                token_ids = input_ids
            else:
                token_ids = input_ids.tolist()

            if len(token_ids) == 0:
                errors += 1
                continue

            # Truncate to max_model_len if needed
            if len(token_ids) > args.max_model_len:
                token_ids = token_ids[:args.max_model_len]

            batch_prompts.append({"prompt_token_ids": token_ids})
            batch_token_ids.append((idx, token_ids))

        if not batch_prompts:
            continue

        try:
            outputs = llm.generate(
                batch_prompts,
                sampling_params,
            )

            for i, output in enumerate(outputs):
                idx, token_ids = batch_token_ids[i]
                try:
                    path = output.kv_transfer_params["hidden_states_path"]
                    obj = example_hidden_states_connector.load_hidden_states(path)

                    hidden_states = obj["hidden_states"]  # [seq_len, num_layers * hidden_size]
                    out_token_ids = obj["token_ids"]  # [seq_len]

                    # Validate
                    if hidden_states.shape[0] != len(token_ids):
                        print(f"  [WARN] sample {idx}: hs len {hidden_states.shape[0]} != tokens {len(token_ids)}, skipping")
                        errors += 1
                        continue

                    # Save
                    out_file = output_path / f"hs_{idx}.safetensors"
                    save_file({
                        "hidden_states": hidden_states.to(torch.bfloat16),
                        "token_ids": out_token_ids.to(torch.long),
                    }, str(out_file))
                    generated += 1

                    # Clean up temp file
                    if os.path.exists(path):
                        os.remove(path)

                except Exception as e:
                    print(f"  [WARN] sample {idx}: {e}")
                    errors += 1

        except Exception as e:
            print(f"  [WARN] batch error: {e}")
            errors += len(batch_indices)

    print(f"\nDone (worker {args.rank}):")
    print(f"  Generated: {generated}")
    print(f"  Skipped (existing): {len(existing & set(my_indices))}")
    print(f"  Errors: {errors}")
    print(f"  Output: {output_path}")

    # Cleanup temp dir
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
