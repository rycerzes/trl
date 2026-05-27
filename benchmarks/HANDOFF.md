# Benchmark Handoff — AsyncGRPO Gap-Closure Validation

## Status: ✅ Smoke test passing (3 steps end-to-end with NCCL weight sync)

---

## What Was Done

### Environment Setup ✅

- vLLM 0.21.0 installed into `/root/trl/.venv` (torch 2.11+cu130, transformers 5.9, peft 0.19.1, trl 1.5.0.dev0)
- Qwen/Qwen3-4B downloaded and cached
- openai/gsm8k dataset cached
- math-verify + latex2sympy2_extended installed
- venv recreated with `uv pip install`

### Validated Working ✅

- **vLLM server** serves Qwen3-4B with all non-default sampling params (top_p, top_k, min_p, repetition_penalty)
- **GRPO baseline** (colocate mode) runs end-to-end: 3 steps in 21s, metrics logged correctly
- **AsyncGRPO trainer** runs end-to-end: 3 steps in 55.6s with weight sync (~8.7s per sync, 8GB over NCCL packed broadcast)
- **NCCL handshake** between trainer and vLLM 0.21 works (warmup allreduce passes)
- **HTTP API** for pause/resume/start_weight_update all return 200 OK
- **Weight sync** transfers 398 parameters (~8GB) via 8 packed NCCL buffers without hanging
- **Training metrics** look correct: ratio≈1.0, kl≈0.0003, generation_tok/s=32→55

### Smoke Test Results

```
weight_sync_time_s: 8.7s (avg per step)
generation_tok_per_s: 32 → 55 (ramping up as buffer fills)
training_tok/s: ~20
3 steps in 55.6s total
```

---

### Bugs Found & Patched ✅

#### Bug 1: vLLM 0.21 API change (start/finish weight update)

The weight update protocol changed from a single `/update_weights` call (0.18) to a 3-step sequence (0.19+):

```
POST /start_weight_update    ← NEW
POST /update_weights
POST /finish_weight_update   ← NEW
```

**Patch applied** in `trl/experimental/async_grpo/async_rollout_worker.py` `send_weights()`.

#### Bug 2: `is_checkpoint_format` in `_weight_update_info`

The `_weight_update_info` dict contained `"is_checkpoint_format": True` which is **not a valid field** for `NCCLWeightTransferUpdateInfo`. When vLLM's GPU worker calls `parse_update_info(update_dict)`, it does `NCCLWeightTransferUpdateInfo(**update_dict)` which raises `TypeError: got an unexpected keyword argument 'is_checkpoint_format'`.

**Fix applied:** Removed `"is_checkpoint_format": True` from `_weight_update_info`. The `is_checkpoint_format` flag is already correctly communicated via the separate `/start_weight_update` call.

#### Bug 3: Missing NCCL P2P guard for PCIe-only hardware

A10G GPUs are PCIe-only (no NVLink). NCCL's P2P/SHM transport uses CUDA IPC which can hang for large broadcasts between separate processes on PCIe-only hardware.

**Fix applied:** Added `_disable_nccl_p2p_if_unavailable()` function that checks NVLink topology via pynvml and sets `NCCL_P2P_DISABLE=1` + `NCCL_SHM_DISABLE=1` when no NVLink is found. Called before `_init_weight_transfer()`.

**Note:** This must also be applied on the vLLM server side via environment variables when launching.

#### Bug 4: PEFT parameter name mismatch (ROOT CAUSE of the NCCL hang)

The trainer iterated `model.base_model.model.named_parameters()` which yields PEFT-wrapped names (902 params total):
- `model.layers.0.self_attn.q_proj.base_layer.weight` (vLLM expects `q_proj.weight`)
- `model.layers.0.self_attn.q_proj.lora_A.default.weight` (vLLM doesn't have this)
- `model.layers.0.self_attn.q_proj.lora_B.default.weight` (vLLM doesn't have this)

vLLM's model only has **398 parameters** with clean names. When the consumer received the first NCCL buffer and called `model.load_weights()`, `AutoWeightsLoader` raised `ValueError` on the unrecognized `base_layer` submodule. The consumer exited, but the producer kept broadcasting → deadlock.

**Fix applied** in `trl/experimental/async_grpo/async_grpo_trainer.py`, using TRL's own established pattern from `vllm_generation.py`:
```python
name = name.replace(".base_layer", "")
if model.prefix in name:  # model.prefix = "lora_"
    continue
```

Applied in both metadata collection (init) and `_streaming_iter()` (runtime). After fix: 398 params sent, names match vLLM exactly.

**How other frameworks handle this:**

| Framework | Pattern |
|-----------|---------|
| **TRL main code** (`vllm_generation.py`) | `.replace(".base_layer", "")` + `if model.prefix in name: continue` |
| **Prime-RL** | `strip_lora_from_state_dict()` — filters `lora_A`/`lora_B` from state dict |
| **OpenRLHF** | Doesn't merge — syncs only LoRA adapters to vLLM's native LoRA support |

---

## Environment (after container restart)

### Verify

```bash
/root/trl/.venv/bin/python -c "
import torch, vllm, transformers, peft, trl
print(f'torch: {torch.__version__}, vllm: {vllm.__version__}')
print(f'transformers: {transformers.__version__}, peft: {peft.__version__}')
print(f'GPUs: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}')
"
```

### Launch vLLM (correct pattern)

**Do NOT use `CUDA_VISIBLE_DEVICES` to isolate vLLM.** The vLLM process must see all GPUs for correct NCCL topology detection. It defaults to GPU 0 with TP=1.

**Use `--gpu-memory-utilization 0.45`** to leave room for NCCL weight transfer buffers (~1GB). With 0.90, the weight transfer OOMs.

```bash
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 VLLM_SERVER_DEV_MODE=1 \
  /root/trl/.venv/bin/vllm serve Qwen/Qwen3-4B \
    --dtype bfloat16 --max-model-len 768 --gpu-memory-utilization 0.45 \
    --logprobs-mode processed_logprobs \
    --weight-transfer-config '{"backend":"nccl"}' --port 8000 \
    --enforce-eager
```

### Trainer

```bash
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 CUDA_VISIBLE_DEVICES=1 \
  /root/trl/.venv/bin/python benchmarks/scripts/async_grpo_main.py
```

---

## Key Files

| File | Purpose |
|------|---------|
| `benchmarks/async_grpo_gap_closure_benchmark.md` | Full benchmark plan |
| `benchmarks/scripts/run_benchmark.sh` | Orchestrator script |
| `benchmarks/scripts/grpo_baseline.py` | GRPO baseline (colocate mode) |
| `benchmarks/scripts/async_grpo_main.py` | AsyncGRPO trainer (server mode) |
| `trl/experimental/async_grpo/async_rollout_worker.py` | **PATCHED** — bugs 1-3 fixed |
| `trl/experimental/async_grpo/async_grpo_trainer.py` | **PATCHED** — bug 4 fixed |

---

## Important Notes

### vLLM version situation

- TRL pins `vllm>=0.12.0,<=0.18.0` but vLLM 0.18 requires torch 2.10 (ABI-incompatible with torch 2.11/2.12)
- vLLM 0.21 works with torch 2.11 and keeps transformers 5.9 + peft 0.19
- The HTTP API for generation (`/v1/completions`) is stable across versions
- The RLHF weight sync API changed (patched above)
- TRL emits advisory warnings ("we only support ≤0.18") — safe to ignore

### GPU memory for weight transfer

vLLM's NCCL weight transfer allocates ~1GB receive buffers on the inference GPU. With `--gpu-memory-utilization 0.90`, the model + KV cache consume nearly all GPU memory, causing OOM during weight transfer. Use `0.45` (or lower) to leave headroom. On production hardware with more VRAM this is less of a concern.

### GPU device placement for NCCL

The official vLLM pattern (from `rlhf_http_nccl.py`):
- vLLM: launched **without** `CUDA_VISIBLE_DEVICES` (sees all GPUs, uses GPU 0)
- Trainer: uses `cuda:{inference_world_size}` with all GPUs visible, OR uses `CUDA_VISIBLE_DEVICES=1` for HF Trainer compatibility

**Never use `CUDA_VISIBLE_DEVICES` for vLLM** when doing NCCL weight transfer. It confuses NCCL topology detection (per vLLM PR #26709).

### GRPO baseline uses colocate mode (not server mode)

GRPOTrainer's server-mode weight sync uses `/init_communicator/` which was removed in vLLM 0.21. Colocate mode avoids this — vLLM runs in-process on the same GPU, no HTTP weight sync needed.

### Never `kill -9` a CUDA process in this container

Use graceful termination or script-level timeouts. SIGKILL leaks GPU memory to PID 1 permanently.

---

## Branch Changes Summary (4 commits + patches)

### Original branch commits (in `trl/experimental/async_grpo/`):

1. **`model_init_kwargs`** — `create_model_from_path(model, **model_init_kwargs)` instead of hardcoded `AutoModelForCausalLM`
2. **Sampling params** — `top_p`, `top_k`, `min_p`, `repetition_penalty` plumbed to config → trainer → rollout worker → HTTP payload
3. **`ProcessorMixin`/`AutoProcessor`** — `AutoProcessor.from_pretrained` + isinstance dispatch for tokenizer extraction
4. **PEFT/LoRA** — `get_peft_model()`, merge/unmerge in `_sync_weight`, `base_model.model` param iteration for weight sync

### Patches applied during benchmark work:

5. **`send_weights()` start/finish calls** — Added `/start_weight_update` and `/finish_weight_update` HTTP calls around the NCCL transfer (required by vLLM >= 0.19)
6. **Removed `is_checkpoint_format` from `_weight_update_info`** — This field is not accepted by `NCCLWeightTransferUpdateInfo` and caused silent parse failure
7. **Added `_disable_nccl_p2p_if_unavailable()`** — Detects PCIe-only topology via pynvml, disables NCCL P2P/SHM transport to avoid broadcast hangs (same pattern as Prime-RL)
8. **PEFT name mapping in weight sync** — Strip `.base_layer` and skip params matching `model.prefix` (`"lora_"`) in both metadata collection and `_streaming_iter()`, using the same pattern as TRL's `vllm_generation.py`

Bugs 5-7 are pre-existing incompatibilities between the async rollout worker (written for vLLM 0.18) and vLLM 0.21 + PCIe hardware. Bug 8 is caused by the branch's PEFT/LoRA commit (#4) not accounting for how vLLM's `model.load_weights()` expects raw model names.
