# Implementation Status: AsyncGRPOTrainer Feature Parity with GRPOTrainer

**Goal:** Fill feature gaps in `AsyncGRPOTrainer` that are already present in `GRPOTrainer`.

**Issue:** https://github.com/huggingface/trl/issues/5831

**Strategy:** Per Quentin's feedback, changes are **decoupled into 4 separate PRs** for easier review.

**Implementation branch (all-in-one):** `feat/async-grpo-parity` — fully implemented, now being split.

---

## PR Strategy & Order

| PR # | Feature | Branch | Status | Depends on |
|------|---------|--------|--------|------------|
| 1 | `model_init_kwargs` | `feat/async-grpo-model-init-kwargs` | 🔲 To split | None |
| 2 | Sampling parameters | `feat/async-grpo-sampling-params` | 🔲 To split | None |
| 3 | ProcessorMixin handling | `feat/async-grpo-processor-mixin` | 🔲 To split | None |
| 4 | PEFT/LoRA support | `feat/async-grpo-peft` | 🔲 To split | PR#1 (uses `create_model_from_path`) |

Order chosen from least → most complex for smoother review flow.

---

## PR 1 — `model_init_kwargs` support

**Branch:** `feat/async-grpo-model-init-kwargs`

**Files:**
- `trl/experimental/async_grpo/async_grpo_config.py`
- `trl/experimental/async_grpo/async_grpo_trainer.py`

**Changes:**
- Add `_VALID_DICT_FIELDS = _BaseConfig._VALID_DICT_FIELDS + ["model_init_kwargs"]`
- Add `model_init_kwargs: dict[str, Any] | str | None` field to config
- Add docstring entry for `model_init_kwargs`
- Replace direct `AutoModelForCausalLM` loading with `create_model_from_path(model, **model_init_kwargs)`
- Enforce `model_init_kwargs["device_map"] = None` (FSDP2-safe)
- Add import: `from trl.trainer.utils import create_model_from_path`

**Why standalone:** Self-contained config + model loading change. No interaction with other features.

---

## PR 2 — Sampling parameters (`top_p`, `top_k`, `min_p`, `repetition_penalty`)

**Branch:** `feat/async-grpo-sampling-params`

**Files:**
- `trl/experimental/async_grpo/async_grpo_config.py`
- `trl/experimental/async_grpo/async_grpo_trainer.py`
- `trl/experimental/async_grpo/async_rollout_worker.py`

**Changes:**
- Add config fields:
  - `top_p` (default `1.0`)
  - `top_k` (default `0`)
  - `min_p` (default `None`)
  - `repetition_penalty` (default `1.0`)
- Add docstring entries for all four fields
- Pass sampling args from trainer → rollout worker constructor
- `AsyncRolloutWorker.__init__` accepts and stores the four params
- `_generate_one_turn` includes them in the vLLM payload (`min_p` conditionally)

**Why standalone:** Straightforward plumbing from config → worker → vLLM. No model loading or weight sync interaction.

---

## PR 3 — ProcessorMixin handling

**Branch:** `feat/async-grpo-processor-mixin`

**Files:**
- `trl/experimental/async_grpo/async_grpo_trainer.py`

**Changes:**
- Use `AutoProcessor.from_pretrained(...)` when `processing_class` is not provided
- Support both tokenizer and processor inputs
- Extract tokenizer when a `ProcessorMixin` is provided
- Normalize padding token on tokenizer (`pad_token ← eos_token` when needed)
- Pass tokenizer (not processor) to:
  - `super().__init__(..., processing_class=tokenizer, ...)`
  - `AsyncRolloutWorker(..., processing_class=tokenizer, ...)`

**Why standalone:** Only touches trainer init flow. Makes VLM-style models (e.g., Qwen3.5-4B which uses `Qwen3_5ForConditionalGeneration`) work without vision inputs.

---

## PR 4 — PEFT/LoRA support

**Branch:** `feat/async-grpo-peft`

**Depends on:** PR#1 (needs `create_model_from_path` for model loading path)

**Files:**
- `trl/experimental/async_grpo/async_grpo_trainer.py`

**Changes:**
- Add `peft_config: PeftConfig | None` parameter to trainer `__init__`
- Validate `peft_config` type and peft availability
- Wrap model with `get_peft_model(...)`
- Enable input grads for gradient checkpointing when needed
- Add import: `from accelerate.utils import is_peft_model`
- Weight metadata collection uses base model params when PEFT is active:
  - `param_source = model.base_model.model if is_peft_model(model) else model`
- `_streaming_iter` streams full/base model params when PEFT is active
- `_sync_weight` merge/unmerge cycle:
  - `merge_adapter()` before weight transfer
  - Transfer merged full weights to rollout worker/vLLM
  - `unmerge_adapter()` after transfer
  - Preserve pause/barrier/resume + model version update flow

**Why last:** Most complex change. Weight sync merge/unmerge cycle needs careful review. Depends on PR#1's model loading refactor.

---

## Splitting Instructions

Each PR branch should be based on upstream `main`:

```bash
git fetch upstream
git checkout upstream/main

# For each PR, create a fresh branch and cherry-pick/extract relevant changes
git checkout -b feat/async-grpo-model-init-kwargs
# ... extract only PR#1 changes from feat/async-grpo-parity

# Exception: PR#4 may be based on PR#1's branch if needed
git checkout feat/async-grpo-model-init-kwargs
git checkout -b feat/async-grpo-peft
# ... add PEFT changes on top
```

After PR#1 merges, rebase PR#4 onto `main`.

---

## PR Description Template

Each PR should reference the issue and explain scope:

```
Closes part of #5831

This PR adds [feature] to `AsyncGRPOTrainer`, matching the existing behavior in `GRPOTrainer`.

## Changes
- ...

## Testing
- ...
```

---

## Explicitly out of scope (unchanged)

- Multimodal inputs in training forward (images/video log-prob computation)
- beta/KL regularization changes
- Liger kernel support for this path
- Additional loss-type variants beyond current GRPO objective
- DeepSpeed support for async path
- Reward model (`str`/`PreTrainedModel`) support beyond callable reward funcs
