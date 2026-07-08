#!/usr/bin/env python3
"""Probe Gemma4-26B-A4B config for EAGLE-3 maiprofile training setup.

Run this ON THE SERVER (where the model / HF cache is reachable). It reads
ONLY config metadata (no weights, no GPU needed) and prints the exact numbers
we need to fill in the training scaffold:

  1. verifier hidden_size / num_hidden_layers / vocab_size / hidden_act
  2. the DEFAULT target-layer-ids that launch_vllm.py + train.py would compute
     ([2, N//2, N-3, N]) so we can decide whether to pin them explicitly
  3. whether the official EAGLE-3 draft's hidden_size matches the verifier
     (train.py hard-requires draft.hidden_size == verifier.hidden_size)
  4. whether this is an MoE config (num_experts / enable_moe_block etc.)

Usage (server, inside the speculators venv or any env with transformers):

    python examples/train/maiprofile/probe_gemma4_26b.py \
        --verifier google/gemma-4-26B-A4B-it \
        --draft /path/to/official/eagle3_speculator   # optional

Send me the full stdout and I'll lock the scaffold's parameters to real values.
"""

import argparse
import json
import sys


def _text_config(cfg):
    """Gemma4 multimodal configs nest the LM under .text_config."""
    return getattr(cfg, "text_config", cfg)


def _dump_known_fields(cfg):
    fields = [
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "vocab_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "max_position_embeddings",
        "rms_norm_eps",
        "hidden_act",
        "hidden_activation",
        "sliding_window",
        # MoE-ish markers (any of these present + set => MoE target)
        "num_experts",
        "num_local_experts",
        "n_routed_experts",
        "num_experts_per_tok",
        "enable_moe_block",
        "moe_intermediate_size",
        "hidden_size_per_layer_input",
    ]
    out = {}
    for f in fields:
        if hasattr(cfg, f):
            out[f] = getattr(cfg, f)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--verifier",
        default="google/gemma-4-26B-A4B-it",
        help="Target/verifier model id or local path (default: %(default)s)",
    )
    ap.add_argument(
        "--draft",
        default="",
        help="(Optional) official EAGLE-3 draft id/path to cross-check hidden_size",
    )
    ap.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoConfig if the repo needs it",
    )
    args = ap.parse_args()

    try:
        from transformers import AutoConfig
    except Exception as e:  # noqa: BLE001
        print(f"[FATAL] cannot import transformers: {e}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print(f"VERIFIER: {args.verifier}")
    print("=" * 70)
    raw = AutoConfig.from_pretrained(
        args.verifier, trust_remote_code=args.trust_remote_code
    )
    has_text_config = hasattr(raw, "text_config")
    cfg = _text_config(raw)

    print(f"has .text_config wrapper : {has_text_config}")
    print("known fields (text/lm config):")
    known = _dump_known_fields(cfg)
    for k, v in known.items():
        print(f"    {k:28}: {v}")

    N = getattr(cfg, "num_hidden_layers", None)
    hidden = getattr(cfg, "hidden_size", None)
    vocab = getattr(cfg, "vocab_size", None)

    # Reproduce launch_vllm.py / train.py default target-layer-id computation.
    if isinstance(N, int) and N > 3:
        default_ids = [2, N // 2, N - 3, N]
    else:
        default_ids = None
    print()
    print("-" * 70)
    print("DERIVED (what launch_vllm.py + train.py would default to)")
    print("-" * 70)
    print(f"    num_hidden_layers (N)      : {N}")
    print(f"    default target-layer-ids   : {default_ids}")
    print("      (= [2, N//2, N-3, N]; last layer appended by --include-last-layer)")
    print(f"    verifier hidden_size       : {hidden}")
    print(f"    verifier vocab_size        : {vocab}")

    # MoE detection heuristic.
    moe_markers = {
        k: known[k]
        for k in (
            "num_experts",
            "num_local_experts",
            "n_routed_experts",
            "num_experts_per_tok",
            "enable_moe_block",
            "moe_intermediate_size",
        )
        if k in known
    }
    is_moe = any(bool(v) for v in moe_markers.values())
    print()
    print(f"    MoE markers present        : {moe_markers or '(none)'}")
    print(f"    => looks like MoE target?  : {is_moe}")
    print(
        "      NOTE: speculators EAGLE-3 draft is an independent dense decoder and\n"
        "      has NO MoE assert on the verifier (only hidden_size must match), so a\n"
        "      MoE verifier is fine. This differs from DeepSpec's gemma4 eagle3/dspark\n"
        "      prototypes which hard-assert `not enable_moe_block`."
    )

    # Optional cross-check against the official EAGLE-3 draft.
    if args.draft:
        print()
        print("=" * 70)
        print(f"OFFICIAL DRAFT: {args.draft}")
        print("=" * 70)
        try:
            draft_raw = AutoConfig.from_pretrained(
                args.draft, trust_remote_code=args.trust_remote_code
            )
            draft_dict = draft_raw.to_dict()
            interesting = {
                k: draft_dict.get(k)
                for k in (
                    "architectures",
                    "algorithm",
                    "draft_vocab_size",
                    "target_vocab_size",
                    "eagle_aux_hidden_state_layer_ids",
                    "verifier",
                )
                if k in draft_dict
            }
            print("draft top-level fields:")
            print(json.dumps(interesting, indent=4, default=str))
            tlc = draft_dict.get("transformer_layer_config", {})
            if tlc:
                print("transformer_layer_config:")
                for k in (
                    "model_type",
                    "hidden_size",
                    "num_hidden_layers",
                    "intermediate_size",
                    "vocab_size",
                ):
                    print(f"    {k:22}: {tlc.get(k)}")
                d_hidden = tlc.get("hidden_size")
                if d_hidden is not None and hidden is not None:
                    match = d_hidden == hidden
                    print()
                    print(
                        f"    draft.hidden_size ({d_hidden}) == "
                        f"verifier.hidden_size ({hidden}) ? -> {match}"
                    )
                    if not match:
                        print(
                            "    [WARN] train.py raises ValueError on mismatch. If you "
                            "train\n    from scratch (no --from-pretrained) this is "
                            "irrelevant; the draft\n    decoder is built to match the "
                            "verifier automatically."
                        )
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] could not read draft config: {e}", file=sys.stderr)

    print()
    print("DONE. Paste this whole output back to finalize the scaffold params.")


if __name__ == "__main__":
    main()
