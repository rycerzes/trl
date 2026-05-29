# AsyncGRPO Gap-Closure Benchmark — Results (Run 1)

This file summarizes the completed benchmark runs in:

- `benchmarks/results/grpo_baseline/`
- `benchmarks/results/async_grpo_main/`

Both runs used **50 steps** (`max_steps=50`) on `Qwen/Qwen3-4B` with the same core training setup (batching, generations, completion length, LoRA rank).

---

## Throughput / Wall-Time Comparison

Source files:

- `benchmarks/results/grpo_baseline/timing.json`
- `benchmarks/results/async_grpo_main/timing.json`

| Variant | wall_time_s | steps_per_sec |
|---|---:|---:|
| GRPO baseline (server mode) | 558.9317424297333 | 0.08945636149173583 |
| AsyncGRPO (main config) | 440.6367256641388 | 0.11347215764786454 |

Computed deltas (AsyncGRPO vs baseline):

- **Wall-time reduction:** `21.164%`
- **Step-throughput increase:** `26.846%`
- **Speedup factor:** `1.268x`

---

## End-of-Run Trainer Summaries

From `run_1.log` final train summary rows:

### GRPO baseline

```python
{'train_runtime': '558.5', 'train_samples_per_second': '0.716', 'train_steps_per_second': '0.09', 'train_loss': '-0.0009589', 'epoch': '0.006691'}
```

### AsyncGRPO

```python
{'train_runtime': '434.9', 'train_samples_per_second': '0.92', 'train_steps_per_second': '0.115', 'train_loss': '0.0206', 'epoch': '1'}
```

---

## Last-Step Reward / Metrics Snippets (Both Runs)

These are from the **last logged step metrics row** in each `run_1.log`.

### GRPO baseline — last-step snippet

```python
{
  'reward': '0.55',
  'reward_std': '0.3594',
  'rewards/accuracy_reward/mean': '0.55',
  'rewards/accuracy_reward/std': '0.3594',
  'completions/mean_length': '236.7',
  'entropy': '0.105',
  'sampling/importance_sampling_ratio/mean': '5.703e-06',
  'step_time': '11.45'
}
```

(Full last-step row is in `benchmarks/results/grpo_baseline/run_1.log`.)

### AsyncGRPO — last-step snippet

```python
{
  'reward': '0.45',
  'reward_std': '0.1661',
  'rewards/accuracy_reward': '0.45',
  'ratio': '0.9361',
  'kl': '0.6947',
  'completions/mean_length': '149.6',
  'generation_tok_per_s': '348.2',
  'training_tok/s': '326.8',
  'weight_sync_time_s': '5.186',
  'queue_wait_time_s': '0.08815'
}
```

(Full last-step row is in `benchmarks/results/async_grpo_main/run_1.log`.)

---

## Notes

- This is a **single-run** comparison (not multi-seed averaged).
- The benchmark objective here was gap-closure validation + throughput check; by that criterion, AsyncGRPO is faster in this run while remaining stable end-to-end.
