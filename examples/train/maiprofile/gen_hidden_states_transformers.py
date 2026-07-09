#!/usr/bin/env python3
"""Generate hidden states for EAGLE-3 training using pure transformers (no vLLM).

Replaces vLLM's buggy `extract_hidden_states` mode with a simple forward pass.
Reads the prepared dataset (arrow), runs each sample through the target model,
extracts hidden states at the specified layers, and saves as safetensors files
compatible with speculators' train.py.

Output format (per file `hs_<idx>.safetensors`):
    "hidden_states": [seq_len, num_layers * hidden_size]  (all target layers concat)
    "token_ids": [seq_len]  (int64)

train.py's `standardize_data_v1` then splits this into:
    hidden_states[:, :-hidden_size] -> draft input (first N-1 layers concat)
    hidden_states[:, -hidden_size:] -> verifier_last_hidden_states (last layer)

Usage (on server, 8x A100):
    python examples/train/maiprofile/gen_hidden_states_transformers.py \
        --model google/gemma-4-26B-A4B-it \
        --prepared-data $EAGLE3/prepared \
        --output $EAGLE3/hidden_states \
        --batch-size 1

Supports resume (skips existing hs_*.safetensors files).
"""

import argparse
import gc
import os
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Target model (HF id or local path)")
    p.add_argument("--prepared-data", required=True, help="Path to prepared arrow dataset")
    p.add_argument("--output", required=True, help="Output dir for hs_*.safetensors")
    p.add_argument("--target-layer-ids", type=int, nargs="+", default=None,
                   help="Layer IDs to extract. Default: [2, N//2, N-3, N]")
    p.add_argument("--batch-size", type=int, default=1, help="Samples per forward pass")
    p.add_argument("--max-samples", type=int, default=None, help="Limit samples (for testing)")
    p.add_argument("--world-size", type=int, default=1, help="Total number of parallel workers")
    p.add_argument("--rank", type=int, default=0, help="This worker's rank (0-indexed)")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"],
                   help="Model dtype")
    p.add_argument("--device-map", default="auto", help="Device map for model loading")
    return p.parse_args()


def get_target_layer_ids(config, explicit_ids=None):
    """Compute default target layer IDs matching launch_vllm.py logic."""
    if hasattr(config, "text_config"):
        config = config.text_config
    N = config.num_hidden_layers
    if explicit_ids:
        return explicit_ids
    # Default: [2, N//2, N-3, N] (N = last hidden layer output)
    return [2, N // 2, N - 3, N]


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # Load config to determine layers
    print(f"Loading config from {args.model}...")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    hidden_size = text_config.hidden_size
    target_layer_ids = get_target_layer_ids(config, args.target_layer_ids)
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_hidden_layers: {text_config.num_hidden_layers}")
    print(f"  target_layer_ids: {target_layer_ids}")
    print(f"  output hidden_states shape: [seq_len, {len(target_layer_ids)} * {hidden_size}]")
    print()

    # Load prepared dataset
    print(f"Loading prepared dataset from {args.prepared_data}...")
    dataset = load_from_disk(args.prepared_data)
    total = len(dataset)
    if args.max_samples:
        total = min(total, args.max_samples)

    # Compute this worker's index range
    all_indices = list(range(total))
    if args.world_size > 1:
        # Each rank gets a contiguous chunk
        chunk_size = (total + args.world_size - 1) // args.world_size
        start = args.rank * chunk_size
        end = min(start + chunk_size, total)
        my_indices = all_indices[start:end]
        print(f"  Worker {args.rank}/{args.world_size}: indices [{start}, {end}) = {len(my_indices)} samples")
    else:
        my_indices = all_indices
    print(f"  Total dataset: {total}, this worker: {len(my_indices)}")

    # Check existing files (resume)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    existing = set()
    for f in output_path.glob("hs_*.safetensors"):
        try:
            idx = int(f.stem.split("_")[1])
            existing.add(idx)
        except (IndexError, ValueError):
            pass
    if existing:
        print(f"  Found {len(existing)} existing files, will skip them (resume)")
    print()

    # Load model
    print(f"Loading model {args.model} ({args.dtype}, device_map={args.device_map})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()
    print("  Model loaded.")
    print()

    # Generate hidden states
    print(f"Generating hidden states for {len(my_indices)} samples...")
    errors = 0
    generated = 0
    for idx in tqdm(my_indices, desc=f"Worker {args.rank}"):
        if idx in existing:
            continue

        sample = dataset[idx]
        input_ids = sample["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        else:
            input_ids = torch.tensor(input_ids.tolist(), dtype=torch.long)

        seq_len = input_ids.shape[0]
        if seq_len == 0:
            errors += 1
            continue

        try:
            with torch.no_grad():
                inputs = input_ids.unsqueeze(0).to(model.device)
                outputs = model(
                    input_ids=inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )

            # Extract hidden states at target layers
            # outputs.hidden_states is a tuple of (num_layers+1) tensors
            # Index 0 = embedding output, index i = layer i output
            all_hidden = outputs.hidden_states  # tuple of [1, seq_len, hidden_size]

            layer_tensors = []
            for layer_id in target_layer_ids:
                # layer_id=N means the last layer output (index N in hidden_states)
                if layer_id >= len(all_hidden):
                    layer_id = len(all_hidden) - 1
                h = all_hidden[layer_id][0].to(dtype=dtype, device="cpu")  # [seq_len, hidden_size]
                layer_tensors.append(h)

            # Concat all layers: [seq_len, num_layers * hidden_size]
            hidden_states = torch.cat(layer_tensors, dim=-1)

            # Save
            out_file = output_path / f"hs_{idx}.safetensors"
            save_file({
                "hidden_states": hidden_states,
                "token_ids": input_ids.to(torch.long),
            }, str(out_file))
            generated += 1

        except torch.cuda.OutOfMemoryError:
            print(f"  [WARN] OOM on sample {idx} (seq_len={seq_len}), skipping")
            torch.cuda.empty_cache()
            gc.collect()
            errors += 1
            continue
        except Exception as e:
            print(f"  [WARN] Error on sample {idx}: {e}")
            errors += 1
            continue

    print(f"\nDone (worker {args.rank}):")
    print(f"  Generated: {generated}")
    print(f"  Skipped (existing): {len(existing & set(my_indices))}")
    print(f"  Errors: {errors}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
