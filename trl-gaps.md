# TRL `AsyncGRPOTrainer` gaps (comprehensive, Async-only)

_Last updated: 2026-05-19_

This file now focuses on **all source-visible gaps specifically related to `trl.experimental.async_grpo.AsyncGRPOTrainer`**, primarily by diffing it against TRL's `GRPOTrainer` API/config surface and checking async worker/runtime constraints.

> Scope note: this is a **comprehensive parity checklist** for AsyncGRPO as implemented in TRL main at time of writing. It is not a list of every feature in every RL framework.

---

## 0) Baseline reality

TRL docs explicitly state AsyncGRPO is intentionally smaller:

```md
Not all features from [`GRPOTrainer`] are available; refer to [`AsyncGRPOConfig`] for the supported parameters.
```

and also:

```md
For distributed training, only FSDP2 is supported (DeepSpeed ZeRO is not).
```

---

## 1) Constructor/API parity gaps (vs `GRPOTrainer`)

`GRPOTrainer.__init__` includes:
- `eval_dataset`
- `reward_processing_classes`
- `peft_config`
- `rollout_func`

`AsyncGRPOTrainer.__init__` does **not** include these.

### Evidence

```python
# trl/trainer/grpo_trainer.py
GRPOTrainer(..., eval_dataset=..., reward_processing_classes=..., peft_config=..., rollout_func=..., ...)
```

```python
# trl/experimental/async_grpo/async_grpo_trainer.py
AsyncGRPOTrainer(..., tools=None, environment_factory=None, rollout_worker=None)
```

### Practical impact

- No first-class eval dataset wiring in async trainer constructor.
- No first-class reward tokenizer wiring for reward models.
- No first-class PEFT/LoRA entry point.
- No pluggable rollout function entry point (async path is tied to async worker + vLLM server flow).

---

## 2) Model input/init flexibility gaps

## 2.1 `model` type is narrower

`AsyncGRPOTrainer` expects `model: str` and loads internally. `GRPOTrainer` supports string or instantiated model / PEFT model object.

## 2.2 Hardcoded fp32 load in AsyncGRPO

### Evidence

```python
# trl/experimental/async_grpo/async_grpo_trainer.py
model = AutoModelForCausalLM.from_pretrained(model, device_map=None, dtype=torch.float32)
```

### Practical impact

- No `model_init_kwargs`-style control (dtype/attn impl/etc.) in async path.
- Higher memory pressure on constrained GPUs (notably 4B on 3xA10G trainer split).

---

## 3) PEFT/LoRA parity gap

AsyncGRPO has no `peft_config` argument and no built-in PEFT wrapping path equivalent to GRPOTrainer.

### Evidence

- `GRPOTrainer` constructor includes `peft_config`.
- `AsyncGRPOTrainer` constructor does not.

### Practical impact

- Async + LoRA requires patching/forking, or using a different framework/path.

---

## 4) Reward stack parity gaps

## 4.1 Narrower reward function type

### Evidence

```python
# async trainer
RewardFunc = Callable[..., list[float]]
```

```python
# grpo trainer
RewardFunc = str | PreTrainedModel | Callable[..., list[float | None]]
```

### Practical impact

- Async path is callable-centric; no first-class reward model id / `PreTrainedModel` reward plumbing like GRPOTrainer.

## 4.2 No `reward_processing_classes`

- GRPO supports tokenizer/processors for reward models.
- AsyncGRPO has no parity hook.

## 4.3 Reward weighting/aggregation controls missing from async config

No async equivalents for:
- `reward_weights`
- `multi_objective_aggregation`
- `scale_rewards`

---

## 5) Loss/objective parity gaps

AsyncGRPO uses a fixed async clipped objective path; many GRPO objective variants are not exposed in async config.

Missing async equivalents include:
- `loss_type` variants (`grpo`, `dr_grpo`, `dapo`, `bnpo`, `cispo`, `sapo`, `luspo`, `vespo`)
- `importance_sampling_level`
- `num_iterations`
- `beta` / ref-policy regularization controls
- `sync_ref_model`, `ref_model_mixup_alpha`, `ref_model_sync_steps`
- `mask_truncated_completions`
- `top_entropy_quantile`
- `off_policy_mask_threshold`
- `use_bias_correction_kl`
- `delta` (two-sided clipping option)
- `sapo_*` and `vespo_*` family knobs

---

## 6) vLLM/off-policy correction parity gaps

AsyncGRPO includes staleness dropping but lacks GRPO's richer mismatch correction config surface.

Missing async equivalents include:
- `vllm_importance_sampling_correction`
- `vllm_importance_sampling_mode`
- `vllm_importance_sampling_cap`

### Evidence (async staleness behavior)

```python
# trl/experimental/async_grpo/async_grpo_trainer.py
staleness = self.model_version_fn() - sample.model_version
if staleness > self.max_staleness:
    continue
```

---

## 7) Generation/sampling control parity gaps

Async worker request payload is minimal (temperature + max_tokens + fixed fields), and many GRPO generation knobs are absent.

### Evidence

```python
# trl/experimental/async_grpo/async_rollout_worker.py
payload = {
  "model": self.model_name,
  "prompt": prompt_ids,
  "max_tokens": self.max_tokens,
  "temperature": self.temperature,
  "n": 1,
  "return_token_ids": True,
  "logprobs": 0,
}
```

Missing async config equivalents include:
- `top_p`, `top_k`, `min_p`
- `repetition_penalty`
- `generation_kwargs`
- `cache_implementation`
- `generation_batch_size`, `steps_per_generation`
- `num_generations_eval`

---

## 8) vLLM integration mode/config parity gaps

AsyncGRPO is tied to external vLLM server semantics; missing GRPO vLLM mode knobs include:
- `use_vllm`
- `vllm_mode` (`server`/`colocate`)
- `vllm_model_impl`
- `vllm_structured_outputs_regex`
- `vllm_server_host`, `vllm_server_port`
- `vllm_group_port`
- `vllm_gpu_memory_utilization`
- `vllm_max_model_length`
- `vllm_tensor_parallel_size`
- `vllm_enable_sleep_mode`

Also, docs require separate vLLM + trainer GPU partition for AsyncGRPO flow.

---

## 9) Distributed/runtime backend gaps

## 9.1 DeepSpeed ZeRO unsupported for AsyncGRPO distributed path

### Evidence (docs)

```md
For distributed training, only FSDP2 is supported (DeepSpeed ZeRO is not).
```

## 9.2 DS3 generation-specific controls absent

No async equivalent for GRPO's `ds3_gather_for_generation`.

---

## 10) Kernel/perf feature gaps

AsyncGRPO explicitly rejects liger in current implementation.

### Evidence

```python
# trl/experimental/async_grpo/async_grpo_trainer.py
if self.args.use_liger_kernel:
    raise NotImplementedError("`use_liger_kernel` is not supported yet.")
```

---

## 11) Tooling gaps for async agentic workloads

Async rollout worker currently forbids async tool functions.

### Evidence

```python
# trl/experimental/async_grpo/async_rollout_worker.py
if inspect.iscoroutinefunction(tool):
    raise ValueError("Asynchronous tools are not supported in AsyncRolloutWorker yet.")
```

---

## 12) Setup/dependency fragility gap

Docs call out a real dependency conflict between required versions of `vllm` and `transformers`.

### Evidence (docs)

```md
pip install 'vllm>=0.17.1'
pip install 'transformers>=5.2.0' --no-deps
```

---

## 13) Full `GRPOConfig` fields missing in `AsyncGRPOConfig` (exhaustive list)

The following **54 fields** exist in `GRPOConfig` but not in `AsyncGRPOConfig`:

- `beta`
- `cache_implementation`
- `cast_lm_head_to_fp32`
- `delta`
- `disable_dropout`
- `ds3_gather_for_generation`
- `generation_batch_size`
- `generation_kwargs`
- `importance_sampling_level`
- `log_completions_hub_repo`
- `log_unique_prompts`
- `loss_type`
- `mask_truncated_completions`
- `min_p`
- `model_init_kwargs`
- `multi_objective_aggregation`
- `num_generations_eval`
- `num_iterations`
- `off_policy_mask_threshold`
- `pad_to_multiple_of`
- `ref_model_mixup_alpha`
- `ref_model_sync_steps`
- `remove_unused_columns`
- `repetition_penalty`
- `reward_weights`
- `sapo_temperature_neg`
- `sapo_temperature_pos`
- `scale_rewards`
- `shuffle_dataset`
- `steps_per_generation`
- `sync_ref_model`
- `top_entropy_quantile`
- `top_k`
- `top_p`
- `use_bias_correction_kl`
- `use_transformers_paged`
- `use_vllm`
- `vespo_k_neg`
- `vespo_k_pos`
- `vespo_lambda_neg`
- `vespo_lambda_pos`
- `vllm_enable_sleep_mode`
- `vllm_gpu_memory_utilization`
- `vllm_group_port`
- `vllm_importance_sampling_cap`
- `vllm_importance_sampling_correction`
- `vllm_importance_sampling_mode`
- `vllm_max_model_length`
- `vllm_mode`
- `vllm_model_impl`
- `vllm_server_host`
- `vllm_server_port`
- `vllm_structured_outputs_regex`
- `vllm_tensor_parallel_size`

---

## 14) Cross-framework context (why users notice these gaps)

Other stacks (VeRL, Prime-RL, Miles/Slime, NeMo-RL, SkyRL) generally expose more first-class controls for one or more of:
- LoRA/PEFT in async RL,
- richer off-policy/staleness/IS controls,
- disaggregated trainer/inference orchestration,
- and broader backend/runtime knobs.

This is why AsyncGRPO users hit patching needs earlier in constrained-hardware or production-like setups.

---

## References

### TRL
- Async docs: https://raw.githubusercontent.com/huggingface/trl/main/docs/source/async_grpo_trainer.md
- Async trainer code: https://raw.githubusercontent.com/huggingface/trl/main/trl/experimental/async_grpo/async_grpo_trainer.py
- Async config: https://raw.githubusercontent.com/huggingface/trl/main/trl/experimental/async_grpo/async_grpo_config.py
- Async worker: https://raw.githubusercontent.com/huggingface/trl/main/trl/experimental/async_grpo/async_rollout_worker.py
- GRPO trainer: https://raw.githubusercontent.com/huggingface/trl/main/trl/trainer/grpo_trainer.py
- GRPO config: https://raw.githubusercontent.com/huggingface/trl/main/trl/trainer/grpo_config.py

### Related framework references used for comparison context
- VeRL docs/repo: https://github.com/verl-project/verl
- Slime repo: https://github.com/THUDM/slime
- Miles repo/docs: https://github.com/radixark/miles
- Prime-RL repo/docs: https://github.com/PrimeIntellect-ai/prime-rl
- NeMo-RL GRPO: https://raw.githubusercontent.com/NVIDIA-NeMo/RL/main/nemo_rl/algorithms/grpo.py
- SkyRL async tutorial: https://docs.skyrl.ai/docs/tutorials/one_step_off_async
- Verifiers (Prime): https://github.com/PrimeIntellect-ai/verifiers
