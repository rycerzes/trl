# GRPO vs AsyncGRPO Benchmark Plan (Gap-Closure Validation)

## Objective

Validate **both correctness and throughput** for the AsyncGRPO implementation in this branch, with a setup that exercises all recently closed gaps:

- `create_model_from_path(...)` model loading path
- `model_init_kwargs` pass-through
- `ProcessorMixin` / `AutoProcessor` handling
- sampling params plumbing to vLLM (`top_p`, `top_k`, `min_p`, `repetition_penalty`)
- PEFT/LoRA integration and weight sync merge/unmerge flow
- Qwen3.5 text-only forward compatibility

This benchmark is intended to validate the implementation work in `trl-implementation-plan.md`, not just raw tokens/sec.

---

## Hardware

- **2x A10 (24GB each)**
  - GPU A: Trainer
  - GPU B: vLLM server

> AsyncGRPO requires trainer and vLLM server on separate CUDA devices.

---

## Primary Benchmark (main evidence)

### Model
- `Qwen/Qwen3.5-4B`

### Dataset
- `openai/gsm8k` (train split)
- Map to fields expected by reward path:
  - `prompt`
  - `solution`

### Reward
- `accuracy_reward` (keep reward simple to avoid masking trainer/generation effects)

### Required config knobs to explicitly exercise new code paths
- `model_init_kwargs` set (e.g. dtype + attention impl)
- non-default sampling params:
  - `top_p != 1.0`
  - `top_k > 0`
  - `min_p` set
  - `repetition_penalty != 1.0`
- PEFT enabled (`peft_config`)
- leave `processing_class=None` in at least one run to trigger `AutoProcessor.from_pretrained(...)`

---

## A/B Matrix

Use identical effective training settings wherever applicable.

1. **GRPOTrainer baseline**
   - `use_vllm=True`, `vllm_mode="server"`
   - same model/dataset/reward/lengths/batch settings

2. **AsyncGRPOTrainer (main)**
   - same workload settings
   - async-specific defaults first (`max_staleness`, `weight_sync_steps`, auto `max_inflight_tasks`)

3. **AsyncGRPOTrainer (tuned, optional)**
   - tune only async control knobs (`weight_sync_steps`, `max_staleness`, optionally `max_inflight_tasks`)
   - keep algorithmic workload unchanged

---

## Suggested Throughput-Oriented Settings

Start here and adjust if memory pressure appears:

- `max_completion_length`: `384` or `512`
- `num_generations`: `8` (drop to 4 if needed)
- prompt length cap: around `256`
- fixed `max_steps`: `200-500`
- run at least `2-3` repeats per config
- ignore first ~`20` steps for warmup in throughput summary

These settings are large enough to make generation cost visible, which is necessary for async overlap to show measurable gains.

---

## Metrics to Report

### Throughput / latency
- wall-clock time
- step time (or equivalent)
- iterations/sec
- tokens/sec (if available)

### Correctness metrics (primary)
- **Run stability:** no crashes, no NaN/Inf loss, no deadlocks/timeouts.
- **Reward learning signal:** compare smoothed `reward` (or `reward/accuracy_reward/mean`) over steps; Async should track GRPO trend and final-window mean.
- **Policy health:** `kl`, `entropy`, and `clip_ratio` should stay in the same regime as GRPO (no collapse/explosion).
- **Generation behavior parity:** `completions/mean_length` and `completions/clipped_ratio` should be comparable to GRPO.
- **Optional held-out check (recommended):** fixed small GSM8K validation slice (e.g. 200 prompts), same decoding settings, compare exact-match/accuracy between checkpoints.

### Async diagnostics (for interpreting correctness/speed)
- stale-sample drop rate (must be bounded; not dominating training)
- `weight_sync_time_s`
- `queue_wait_time_s`
- `wait_scoring_ms`
- `generation_tok_per_s`

---

## Success Criteria

1. **Correctness parity (trend-level)**
   - Async run is stable (no NaN/crash/hang) and follows similar reward/KL/entropy trends to GRPO.
   - Final-window reward mean is close to GRPO (target: within ~5-10% relative gap for short runs).
   - If held-out eval is used, accuracy gap should be small (target: within ~1-3 absolute points).

2. **Speedup present**
   - Async shows positive throughput gain vs GRPO server-mode baseline on the same model/workload.

3. **Gap-closure coverage confirmed**
   - Logs/configs demonstrate that all intended new paths were actually exercised.

---

## Expected Results (practical)

On 2x A10 with 4B models, a realistic async speedup expectation is typically in a **moderate** range (often around ~`1.1x` to `1.6x`, workload-dependent), not multi-node headline gains.

---

## Notes

- Large public claims (e.g. Red Hat / multi-node H100 comparisons) are useful directional context, but this branch validation should be based on **same-codebase, same-hardware, fair A/B**.
- Use this plan to support both implementation correctness and performance claims in one benchmark package.
