#!/usr/bin/env python3
"""Regenerate assistant responses for MAI Profile raw data via vLLM Chat API.

Reads raw_data/<layer>.jsonl files (with `prompt_messages` field), sends the
system+user prompt to the 26B-A4B target model through vLLM's OpenAI-compatible
chat completions endpoint, and writes a conversations JSONL ready for
speculators' prepare_data.py.

Output schema (per line):
{
  "id": "<layer>:<prompt_hash>",
  "source_layer": "<layer>",
  "user_id": "...",
  "prompt_hash": "...",
  "conversations": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<generated>"}
  ]
}

This matches what speculators' preprocessing expects: the conversations field
with role/content, including the assistant turn for loss mask creation.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import aiohttp
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    # Input: EITHER raw-dir + layers, OR a pre-split jsonl file (from DSpark's
    # prepare_maiprofile_splits.py). Use --input-file when you want to preserve
    # the DSpark train/eval split and only regenerate the train portion.
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--raw-dir", help="Directory with raw <layer>.jsonl files (regenerates everything)")
    input_group.add_argument("--input-file", help="Pre-split train JSONL (from prepare_maiprofile_splits.py); preserves DSpark train/eval split")
    p.add_argument("--layers", default=None, help="Comma-separated layer names (required with --raw-dir, ignored with --input-file)")
    p.add_argument("--outfile", required=True, help="Output JSONL path")
    p.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default=None, help="Model name (auto-detected if omitted)")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--limit", type=int, default=None, help="Stop after N samples (for pilot runs)")
    p.add_argument("--resume", action="store_true", help="Skip IDs already in outfile")
    args = p.parse_args()
    if args.raw_dir and not args.layers:
        p.error("--layers is required when using --raw-dir")
    return args


def load_records(raw_dir: str, layers: list[str]):
    """Yield (record_dict, layer) from all specified layer files."""
    raw_path = Path(raw_dir)
    for layer in layers:
        filepath = raw_path / f"{layer}.jsonl"
        if not filepath.exists():
            print(f"[WARN] {filepath} not found, skipping layer", file=sys.stderr)
            continue
        if filepath.stat().st_size == 0:
            print(f"[WARN] {filepath} is empty, skipping layer", file=sys.stderr)
            continue
        with filepath.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[WARN] {layer} line {line_num}: invalid JSON, skipping", file=sys.stderr)
                    continue
                # Normalize: raw data uses prompt_messages
                messages = record.get("prompt_messages") or record.get("conversations") or []
                if not isinstance(messages, list) or not messages:
                    continue
                # Must have at least one user turn
                if not any(
                    isinstance(m, dict) and m.get("role") == "user"
                    for m in messages
                ):
                    continue
                yield {
                    "id": f"{layer}:{record.get('prompt_hash') or line_num}",
                    "source_layer": layer,
                    "user_id": record.get("user_id"),
                    "prompt_hash": record.get("prompt_hash"),
                    "messages": [
                        {"role": m["role"], "content": m.get("content") or ""}
                        for m in messages
                        if isinstance(m, dict) and m.get("role") in ("system", "user")
                    ],
                }, layer


def load_records_from_split(input_file: str):
    """Yield (record_dict, layer) from a pre-split train JSONL.

    The split file (produced by prepare_maiprofile_splits.py) already has
    `conversations` in role/content format. We strip any existing assistant
    turns (they were from the 12B target) and regenerate with our 26B-A4B.
    """
    filepath = Path(input_file)
    if not filepath.exists():
        print(f"[FATAL] input file not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    with filepath.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] line {line_num}: invalid JSON, skipping", file=sys.stderr)
                continue
            conversations = record.get("conversations") or record.get("prompt_messages") or []
            if not isinstance(conversations, list) or not conversations:
                continue
            # Keep only system + user (drop assistant from prior regen)
            messages = [
                {"role": m["role"], "content": m.get("content") or ""}
                for m in conversations
                if isinstance(m, dict) and m.get("role") in ("system", "user")
            ]
            if not any(m["role"] == "user" for m in messages):
                continue
            layer = record.get("source_layer") or "unknown"
            yield {
                "id": record.get("id") or f"{layer}:{record.get('prompt_hash') or line_num}",
                "source_layer": layer,
                "user_id": record.get("user_id"),
                "prompt_hash": record.get("prompt_hash"),
                "messages": messages,
            }, layer


def load_seen(path: str) -> set[str]:
    seen = set()
    if not os.path.isfile(path):
        return seen
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("id"):
                    seen.add(obj["id"])
            except json.JSONDecodeError:
                pass
    return seen


async def detect_model(endpoint: str) -> str:
    models_url = endpoint.replace("/v1/chat/completions", "/v1/models")
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(models_url) as resp:
            data = await resp.json()
            models = data.get("data", [])
            if models:
                name = models[0]["id"]
                print(f"Auto-detected model: {name}")
                return name
            raise ValueError("No models found")


async def worker(
    session: aiohttp.ClientSession,
    queue: asyncio.Queue,
    args,
    out_fh,
    err_fh,
    progress,
    stats: dict[str, int],
):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        record = item["record"]
        messages = record["messages"]
        start = time.time()

        try:
            payload = {
                "model": args.model,
                "messages": messages,
                "max_tokens": args.max_tokens,
            }
            async with session.post(args.endpoint, json=payload) as resp:
                if not resp.ok:
                    body = (await resp.text())[:300]
                    raise RuntimeError(f"HTTP {resp.status}: {body}")
                data = await resp.json()

            choice = data["choices"][0]
            assistant_content = choice["message"].get("content")
            if not assistant_content:
                raise ValueError("empty assistant content")

            # Build output conversations (role/content schema for speculators)
            conversations = list(messages) + [
                {"role": "assistant", "content": assistant_content}
            ]
            output = {
                "id": record["id"],
                "source_layer": record["source_layer"],
                "user_id": record.get("user_id"),
                "prompt_hash": record.get("prompt_hash"),
                "conversations": conversations,
            }
            out_fh.write(json.dumps(output, ensure_ascii=False) + "\n")
            out_fh.flush()
            stats["ok"] += 1

        except Exception as e:  # noqa: BLE001
            error_out = {
                "id": record["id"],
                "source_layer": record["source_layer"],
                "error": repr(e),
                "latency_s": round(time.time() - start, 3),
            }
            err_fh.write(json.dumps(error_out, ensure_ascii=False) + "\n")
            err_fh.flush()
            stats["errors"] += 1

        finally:
            progress.set_postfix(ok=stats["ok"], err=stats["errors"], refresh=False)
            progress.update(1)
            queue.task_done()


async def main():
    args = parse_args()

    if args.model is None:
        args.model = await detect_model(args.endpoint)
    print(f"Model: {args.model}")

    # Load records from either raw layers or pre-split file
    if args.input_file:
        print(f"Input: {args.input_file} (pre-split, preserving DSpark train/eval split)")
        records = list(load_records_from_split(args.input_file))
    else:
        layers = [l.strip() for l in args.layers.split(",") if l.strip()]
        print(f"Layers: {layers}")
        records = list(load_records(args.raw_dir, layers))

    print(f"Output: {args.outfile}")
    print(f"Total records: {len(records)}")
    print()

    # Resume support
    seen = load_seen(args.outfile) if args.resume else set()
    if seen:
        before = len(records)
        records = [(r, l) for r, l in records if r["id"] not in seen]
        print(f"Resuming: skipped {before - len(records)} already-done, {len(records)} remaining")

    # Limit (for pilot runs)
    if args.limit and len(records) > args.limit:
        records = records[: args.limit]
        print(f"Limiting to {args.limit} samples (pilot mode)")

    if not records:
        print("Nothing to do.")
        return

    # Error file
    base, ext = os.path.splitext(args.outfile)
    error_file = f"{base}.errors{ext or '.jsonl'}"

    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=90, sock_read=None)
    connector = aiohttp.TCPConnector(limit=None, force_close=False)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        with (
            open(args.outfile, "a", encoding="utf-8") as out_fh,
            open(error_file, "a", encoding="utf-8") as err_fh,
            tqdm(total=len(records), desc="Regenerating", unit="sample") as progress,
        ):
            stats = {"ok": 0, "errors": 0}
            workers = [
                asyncio.create_task(
                    worker(session, queue, args, out_fh, err_fh, progress, stats)
                )
                for _ in range(args.concurrency)
            ]

            for record, _ in records:
                await queue.put({"record": record})

            for _ in workers:
                await queue.put(None)
            await asyncio.gather(*workers)

    print(f"\nDone: {stats['ok']} ok, {stats['errors']} errors")
    print(f"Output: {args.outfile}")
    if stats["errors"]:
        print(f"Errors: {error_file}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
