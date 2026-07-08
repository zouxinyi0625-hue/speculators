# EAGLE-3 × MAI Profile — Gemma4-26B-A4B draft training

Train an EAGLE-3 draft (speculator) for `google/gemma-4-26B-A4B-it` on **MAI
Profile** data, using the `vllm-project/speculators` framework. Goal: beat the
current MTP baseline (online **2113 tok/s** / offline **2023** / accept_len
**5.03**) by **+10–20%** on the same data + concurrency.

> **Discipline** (per project rules): the local machine is dev-only (no usable
> GPU). These scripts are written to be *pulled and run on the 8×A100 server*.
> No run numbers are ever fabricated — accept rate / tok/s come only from real
> server runs. Everything below is derived from the speculators source, not
> guessed.

---

## Why this reuses the DSpark data

The DSpark line already produced a MAI Profile **short-layer prompt split** in
the exact `conversations` jsonl schema that speculators' `prepare_data.py`
accepts for custom data:

```
$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/prepared_prompts/20260615/short_layers/train_maiprofile_short_layers.jsonl
```

EAGLE-3 **online** training only needs the *prompts*: the assistant responses
and target hidden states are generated on-the-fly by the live vLLM verifier.
So — unlike DSpark — we do **not** need the `regenerated/` or `target_cache/`
artifacts. One less thing to build.

---

## Prereqs — two venvs (keep separate)

```bash
# speculators venv — data prep + training
uv venv speculators_venv && source speculators_venv/bin/activate
uv pip install -e .          # from the repo root (this fork)

# vLLM venv — serve the verifier + extract hidden states
uv venv vllm_venv && source vllm_venv/bin/activate
uv pip install "vllm>=0.18"
```

Install your experiment tracker (tensorboard/wandb/…) in the **speculators** venv.

---

## Step 0 (do this first): probe the real 26B-A4B config

The local box can't read the 26B-A4B config, so several numbers below are
**TBD until you run this on the server** and paste the output back:

```bash
# any env with transformers is fine (no GPU needed)
python examples/train/maiprofile/probe_gemma4_26b.py \
    --verifier google/gemma-4-26B-A4B-it
```

It prints: `num_hidden_layers` (N), `hidden_size`, `vocab_size`, the **default
target-layer-ids** `[2, N//2, N-3, N]`, and MoE markers. We use these to decide
whether to pin `--target-layer-ids` explicitly.

---

## The three steps

All commands run **on the server**, from the repo root.

### Step 1 — Prepare data (speculators venv)

```bash
bash examples/train/maiprofile/prepare_maiprofile_eagle3.sh
```

Wraps `scripts/prepare_data.py`. Produces `output/maiprofile_eagle3_26b/` with
arrow shards + `token_freq.pt`. Override `DATE`, `MAX_SAMPLES`, `SEQ_LENGTH`,
`OUTPUT_DIR` via env.

### Step 2 — Launch vLLM verifier (vLLM venv, keep running)

```bash
bash examples/train/maiprofile/launch_vllm_gemma4_26b.sh
# print-only:  DRY_RUN=1 bash examples/train/maiprofile/launch_vllm_gemma4_26b.sh
```

Wraps `scripts/launch_vllm.py` with `method=extract_hidden_states` + a KV
connector that writes hidden states to `HIDDEN_STATES_PATH`. Defaults to 4 GPUs
(`0,1,2,3`) via data parallelism.

### Step 3 — Train (speculators venv, second terminal)

```bash
bash examples/train/maiprofile/train_eagle3_maiprofile.sh
```

Wraps `scripts/train.py` in online mode (`--on-missing generate --on-generate
delete`) via `torchrun` on the other 4 GPUs (`4,5,6,7`). Checkpoints land in
`.../checkpoints/`, with `checkpoint_best` symlinked to the lowest val-loss epoch.

> **GPU split**: online training needs disjoint GPUs for vLLM vs training.
> Default here = 4 (vLLM) + 4 (train). Tune on the server; if training is bursty
> / starved, give vLLM more GPUs (see the tutorial's "inconsistent utilization").

---

## Key parameters explained

These are the ones that actually matter for EAGLE-3. Current scaffold uses
**speculators official defaults** (your call: get it running first, tune later).

| Param (script var) | Default | What it means |
|---|---|---|
| `--draft-vocab-size` (`DRAFT_VOCAB_SIZE`) | **32000** | EAGLE-3 prunes the target's huge vocab (262k for Gemma4) down to a small **draft vocab**. Only the most frequent 32k tokens (from `token_freq.pt`) get their own draft logits; the mapping tables `d2t.npy`/`t2d.npy` translate between draft-id and target-id. Smaller draft vocab = cheaper/faster draft head. Requires `token_freq.pt` from Step 1 — without it this is a no-op and the full vocab is used. |
| `--num-layers` (`NUM_LAYERS`) | **1** | Number of decoder layers in the draft. EAGLE-3's draft is a small independent llama-style decoder; 1 layer is the standard/default (cheap draft = higher end-to-end speedup). More layers can raise acceptance but cost more per draft step. |
| `--ttt-steps` (`TTT_STEPS`) | **3** | **Training-Time Test** unroll depth. During training the draft's *own* prediction is fed back as the next input for this many steps, so at inference the draft is robust to consuming either the target's fused feature `g` or its own previous token `a`. Prevents error accumulation across the K drafted tokens. (Note: EAGLE-3 paper / DeepSpec often use **7**; we start at the speculators default 3 and can raise it.) |
| `--ttt-step-loss-decay` (`TTT_DECAY`) | **1.0** | Per-TTT-step loss weighting. 1.0 = every unrolled step weighted equally. <1.0 down-weights later (harder, further-out) steps. |
| `--target-layer-ids` (`TARGET_LAYER_IDS`) | auto `[2, N//2, N-3, N]` | Which verifier layers' hidden states get fused as the draft's input (EAGLE-3's **multi-layer feature fusion**: low + mid + high + last). **Must be identical in Step 2 and Step 3.** Empty here = both use the same computed default, so they're consistent. Pin explicitly only if you deviate. |
| `--total-seq-len` (`SEQ_LEN`) | **8192** | Max training sequence length. Must be ≥ the `--seq-length` used in Step 1. Lower it (e.g. 4096) if you OOM. MAI Profile short layers are mostly ≤4k, so 4096 may be plenty + faster. |
| `--epochs` / `--lr` | **20 / 1e-4** | train.py defaults. Paper used lr 5e-5; adjust if loss plateaus (see tutorial troubleshooting). |
| `--loss-fn` | **kl_div** | EAGLE-3 trains the draft to match the target's next-token distribution (soft distillation), not hard CE — this is the key lever for acceptance. Left at default. |
| `--on-missing generate` / `--on-generate delete` | — | Online mode: generate hidden states from the live vLLM server on demand, delete after use (save disk). Switch `--on-generate cache` for hybrid (cache epoch 1, reuse later). |

### Why EAGLE-3 has no MoE problem here (unlike DeepSpec)

`speculators`' `Eagle3DraftModel` is an **independent dense decoder** that only
*reads* aux hidden states from the verifier (`eagle_aux_hidden_state_layer_ids`).
It has **no assert on whether the verifier is MoE** — the only hard requirement
is `draft.hidden_size == verifier.hidden_size` (auto-satisfied when training
from scratch, since the draft decoder is built from the verifier config). This
is why official `RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3` exists and why
26B-A4B (MoE) trains fine on this path. Contrast: DeepSpec's
`gemma4/modeling.py` hard-asserts `not enable_moe_block` (dense-only prototype).

---

## Fair comparison vs the MTP baseline

When you benchmark the trained draft (reuse the `vllm-msn` bench scaffold), hold
**all** of these constant vs the MTP baseline, change only draft+method:

- same target **and same quantization** (baseline is FP8 26B — decide explicitly
  whether to serve EAGLE-3 on the FP8 target or bf16; note the draft is trained
  on bf16 hidden states)
- same dataset (`sc1_delta_v2.jsonl` for the throughput bench), same
  `num_prompts`, same output-len, same concurrency
- same `k`: baseline `spec_tokens=5` → set EAGLE-3 `num_speculative_tokens=5`
  (an official EAGLE-3 card may quote k=3; override to 5 for apples-to-apples)

Record **tok/s + per-layer accept_len + per-position acceptance** (per the
roadmap success criteria), not just aggregate tok/s.

---

## Files

| File | Role |
|---|---|
| `probe_gemma4_26b.py` | Server-side config probe (run first; fills in TBDs) |
| `prepare_maiprofile_eagle3.sh` | Step 1 — wraps `prepare_data.py` on the maiprofile split |
| `launch_vllm_gemma4_26b.sh` | Step 2 — wraps `launch_vllm.py` (verifier + hidden states) |
| `train_eagle3_maiprofile.sh` | Step 3 — wraps `train.py` (online, torchrun, 4 GPUs) |

## Open TBDs (resolve on server)

1. Real `num_hidden_layers` / `hidden_size` / default target-layer-ids for
   26B-A4B → run `probe_gemma4_26b.py`.
2. GPU split (4+4 vs other) — tune for datagen vs train balance.
3. `SEQ_LEN` — 4096 likely enough for maiprofile short layers (faster) vs 8192.
4. Whether to bump `TTT_STEPS` 3→7 to match the EAGLE-3 paper once the pipeline
   is verified working.
