#!/usr/bin/env python3
"""Split ALL maiprofile raw layers into train/eval for EAGLE-3 training.

Reads raw_data/<layer>.jsonl, does a seeded shuffle + hold-out eval split per
layer, then writes:
  - train_all_layers.jsonl  (merged train, shuffled)
  - eval_datasets/maiprofile_<layer>.jsonl  (per-layer eval)
  - split_summary.json

Output goes to $AZURE_ML_INPUT_msndni/.../eagle3/<date>/ by default.
Same split logic as DSpark's prepare_maiprofile_splits.py (same seed = same
split), but extended to ALL layers (not just short 5).

Usage (server):
    python examples/train/maiprofile/split_maiprofile_eagle3.py

    # Or override paths:
    python examples/train/maiprofile/split_maiprofile_eagle3.py \
        --input-dir /path/to/raw_data/20260615 \
        --output-dir /path/to/eagle3/20260615
"""

import argparse
import json
import os
import random
from pathlib import Path

ALL_LAYERS = [
    "layer1_actual",
    "layer1_delta",
    "layer1_intent",
    "layer2_coarse_interest",
    "layer2_temporal",
    "layer3_commercial_interests",
    "layer3_persona",
    "layer3_seasonality",
    "layer4_biography",
    "layer4_commercial_preference",
]


def get_msndni_mount() -> Path:
    mount = os.environ.get("AZURE_ML_INPUT_msndni")
    if not mount:
        raise SystemExit(
            "AZURE_ML_INPUT_msndni is not set. Export it or pass --input-dir/--output-dir."
        )
    return Path(mount)


def normalize_messages(record):
    """Extract system+user messages from raw record (prompt_messages or conversations)."""
    messages = record.get("prompt_messages") or record.get("conversations") or []
    if not isinstance(messages, list):
        return []
    out = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if not isinstance(role, str):
            continue
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        if role not in ("system", "user", "assistant"):
            continue
        out.append({"role": role, "content": content})
    return out


def read_layer(input_dir: Path, layer: str):
    path = input_dir / f"{layer}.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = normalize_messages(record)
            if not messages:
                continue
            if not any(m["role"] == "user" for m in messages):
                continue
            yield {
                "id": f"{layer}:{record.get('prompt_hash') or line_num}",
                "source_layer": layer,
                "source_file": path.name,
                "source_line": line_num,
                "user_id": record.get("user_id"),
                "prompt_hash": record.get("prompt_hash"),
                "conversations": messages,
            }


def prompt_text_for_eval(messages):
    system_parts = []
    user_parts = []
    for m in messages:
        if m["role"] == "system":
            system_parts.append(m["content"])
        elif m["role"] == "user":
            user_parts.append(m["content"])
    parts = []
    if system_parts:
        parts.append("\n\n".join(system_parts).strip())
    if user_parts:
        parts.append("\n\n".join(user_parts).strip())
    return "\n\n".join(p for p in parts if p)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--layers", default=",".join(ALL_LAYERS),
                        help="Comma-separated layers (default: all 11)")
    parser.add_argument("--eval-size", type=int, default=200,
                        help="Eval samples per layer")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=980406,
                        help="Same seed as DSpark split for reproducibility")
    parser.add_argument("--date", default="20260615")
    args = parser.parse_args()

    if args.input_dir:
        input_dir = Path(args.input_dir).resolve()
    else:
        input_dir = get_msndni_mount() / "shares/users/zxy/maiprofile/raw_data" / args.date

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = get_msndni_mount() / "shares/users/zxy/maiprofile/eagle3" / args.date

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = output_dir / "eval_datasets"
    eval_dir.mkdir(parents=True, exist_ok=True)

    layers = [l.strip() for l in args.layers.split(",") if l.strip()]
    rng = random.Random(args.seed)
    train_records = []
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "layers": layers,
        "eval_size_per_layer": args.eval_size,
        "seed": args.seed,
        "by_layer": {},
    }

    for layer in layers:
        records = list(read_layer(input_dir, layer) or [])
        rng.shuffle(records)
        eval_records = records[: args.eval_size]
        train_layer = records[args.eval_size:]
        train_records.extend(train_layer)

        # Write per-layer eval
        eval_path = eval_dir / f"maiprofile_{layer}.jsonl"
        with eval_path.open("w", encoding="utf-8") as f:
            for rec in eval_records:
                f.write(json.dumps({
                    "id": rec["id"],
                    "source_layer": rec["source_layer"],
                    "user_id": rec.get("user_id"),
                    "prompt_hash": rec.get("prompt_hash"),
                    "messages": rec["conversations"],
                    "turns": [prompt_text_for_eval(rec["conversations"])],
                }, ensure_ascii=False) + "\n")

        summary["by_layer"][layer] = {
            "raw_records": len(records),
            "train_records": len(train_layer),
            "eval_records": len(eval_records),
        }
        print(f"[{layer}] raw={len(records)} train={len(train_layer)} eval={len(eval_records)}")

    # Merge + shuffle train
    rng.shuffle(train_records)
    if args.max_train_samples is not None:
        train_records = train_records[: args.max_train_samples]

    train_path = output_dir / "train_all_layers.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for rec in train_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary["train_all_file"] = str(train_path)
    summary["train_all_records"] = len(train_records)
    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDone:")
    print(f"  train: {train_path} ({len(train_records)} records)")
    print(f"  eval:  {eval_dir}/ ({args.eval_size} per layer × {len(layers)} layers)")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
