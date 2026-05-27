# AsyncGRPO vs GRPO Benchmark: Gap-Closure Validation

## Objective

Validate **correctness and throughput** for the `AsyncGRPOTrainer` on branch `feat/async-grpo-parity`, exercising all recently closed implementation gaps:

| Gap | How exercised |
|-----|---------------|
| `create_model_from_path(...)` model loading | Pass model as string path |
| `model_init_kwargs` pass-through | Explicit `dtype` + `attn_implementation` |
| `ProcessorMixin` / `AutoProcessor` handling | Leave `processing_class=None` → triggers `AutoProcessor.from_pretrained` |
| Sampling params plumbing (`top_p`, `top_k`, `min_p`, `repetition_penalty`) | All set to non-default values |
| PEFT/LoRA integration & weight sync merge/unmerge | `peft_config=LoraConfig(...)` |

> **Note on Qwen3.5**: TRL pins `vllm>=0.12.0,<=0.18.0`, but Qwen3.5's hybrid linear-attention
> architecture requires vLLM 0.19+. Since all 4 branch commits (`model_init_kwargs`,
> sampling params, `AutoProcessor`/`ProcessorMixin`, PEFT) are model-agnostic infrastructure,
> we use **Qwen3-4B** which is fully supported by vLLM 0.18. Qwen3.5 transformers-side
> compatibility is already validated by mainline CI via tiny-Qwen3.5 fixture tests.

---

## Hardware

| GPU | Role | CUDA device |
|-----|------|-------------|
| NVIDIA A10G (24 GB) | Trainer | `CUDA_VISIBLE_DEVICES=0` |
| NVIDIA A10G (24 GB) | vLLM server | `CUDA_VISIBLE_DEVICES=1` |

---

## Implementation

### Directory layout

```
benchmarks/
├── async_grpo_gap_closure_benchmark.md   ← this file
├── scripts/
│   ├── run_benchmark.sh                  ← orchestrator (launches vLLM + trainer)
│   ├── grpo_baseline.py                  ← GRPOTrainer (sync, vLLM server mode)
│   ├── async_grpo_main.py                ← AsyncGRPOTrainer (default async knobs)
│   ├── async_grpo_tuned.py              ← AsyncGRPOTrainer (tuned async knobs)
│   ├── reward.py                         ← accuracy_reward wrapper
│   └── eval_checkpoint.py               ← optional held-out GSM8K accuracy eval
└── results/                              ← generated after runs
    ├── grpo_baseline/
    ├── async_grpo_main/
    └── async_grpo_tuned/
```

---

## Shared Training Configuration

All runs use identical algorithmic settings:

```python
# --- Model ---
MODEL_ID = "Qwen/Qwen3-4B"
MODEL_INIT_KWARGS = {
    "dtype": "bfloat16",
    "attn_implementation": "flash_attention_2",
}

# --- Dataset ---
DATASET = "openai/gsm8k"
DATASET_SPLIT = "train"
# Map: question → prompt (chat format), answer → solution (after "####")

# --- Reward ---
REWARD_FUNC = accuracy_reward  # from trl.rewards

# --- Generation (non-default sampling to exercise plumbing) ---
MAX_COMPLETION_LENGTH = 384
NUM_GENERATIONS = 8
TEMPERATURE = 0.9
TOP_P = 0.95
TOP_K = 50
MIN_P = 0.05
REPETITION_PENALTY = 1.1

# --- Training ---
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
MAX_STEPS = 300
LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.03
GRADIENT_CHECKPOINTING = True
BF16 = True
SAVE_STRATEGY = "steps"
SAVE_STEPS = 100
LOGGING_STEPS = 5

# --- PEFT/LoRA ---
PEFT_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# --- processing_class ---
# Left as None in all runs → AutoProcessor.from_pretrained triggers

# --- Chat template ---
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
```

---

## A/B Run Matrix

### Run 1: GRPOTrainer baseline (sync, vLLM server mode)

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    output_dir="benchmarks/results/grpo_baseline",
    use_vllm=True,
    vllm_mode="server",
    vllm_server_base_url="http://localhost:8000",
    # All shared settings above
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    max_completion_length=384,
    num_generations=8,
    temperature=0.9,
    top_p=0.95,
    top_k=50,
    min_p=0.05,
    repetition_penalty=1.1,
    max_steps=300,
    learning_rate=1e-5,
    warmup_ratio=0.03,
    gradient_checkpointing=True,
    bf16=True,
    save_strategy="steps",
    save_steps=100,
    logging_steps=5,
    chat_template_kwargs={"enable_thinking": False},
    log_completions=True,
    report_to="none",
)

trainer = GRPOTrainer(
    model=MODEL_ID,
    args=config,
    train_dataset=dataset,
    reward_funcs=accuracy_reward,
    peft_config=PEFT_CONFIG,
    model_init_kwargs=MODEL_INIT_KWARGS,
    # processing_class=None  (default, triggers AutoProcessor)
)
trainer.train()
```

### Run 2: AsyncGRPOTrainer (default async knobs)

```python
from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer

config = AsyncGRPOConfig(
    output_dir="benchmarks/results/async_grpo_main",
    vllm_server_base_url="http://localhost:8000",
    # Async defaults
    max_staleness=4,
    weight_sync_steps=1,
    max_inflight_tasks=-1,  # auto
    # All shared settings
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    max_completion_length=384,
    num_generations=8,
    temperature=0.9,
    top_p=0.95,
    top_k=50,
    min_p=0.05,
    repetition_penalty=1.1,
    max_steps=300,
    learning_rate=1e-5,
    warmup_ratio=0.03,
    gradient_checkpointing=True,
    bf16=True,
    save_strategy="steps",
    save_steps=100,
    logging_steps=5,
    chat_template_kwargs={"enable_thinking": False},
    log_completions=True,
    report_to="none",
)

trainer = AsyncGRPOTrainer(
    model=MODEL_ID,
    args=config,
    train_dataset=dataset,
    reward_funcs=accuracy_reward,
    peft_config=PEFT_CONFIG,
    model_init_kwargs=MODEL_INIT_KWARGS,
    # processing_class=None  (default, triggers AutoProcessor)
)
trainer.train()
```

### Run 3: AsyncGRPOTrainer (tuned async knobs)

```python
config = AsyncGRPOConfig(
    output_dir="benchmarks/results/async_grpo_tuned",
    vllm_server_base_url="http://localhost:8000",
    # Tuned async knobs
    max_staleness=6,
    weight_sync_steps=2,
    max_inflight_tasks=48,  # explicit cap
    # All other shared settings identical to Run 2
    ...
)
```

---

## vLLM Server Launch

Using vLLM 0.21.0 (installed via `uv pip install "vllm==0.21.0" "transformers>=5.9"`).
The trainer merges LoRA weights before syncing to vLLM (full weights, not adapters), so `--enable-lora` is **not** needed.

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_SERVER_DEV_MODE=1 /root/trl/.venv/bin/vllm serve Qwen/Qwen3-4B \
    --dtype bfloat16 \
    --max-model-len 768 \
    --gpu-memory-utilization 0.90 \
    --logprobs-mode processed_logprobs \
    --weight-transfer-config '{"backend":"nccl"}' \
    --port 8000
```

> `--max-model-len 768` = prompt budget (~384) + completion budget (384). Adjust if prompts are longer.

---

## Orchestration Script (`run_benchmark.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../results"
VLLM_PORT=8000
MODEL="Qwen/Qwen3-4B"
NUM_REPEATS=3

# --- Helper ---
wait_for_vllm() {
    echo "Waiting for vLLM server on port ${VLLM_PORT}..."
    for i in $(seq 1 120); do
        if curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo "vLLM server ready."
            return 0
        fi
        sleep 2
    done
    echo "ERROR: vLLM server did not start within 240s"
    return 1
}

kill_vllm() {
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 5
}

launch_vllm() {
    kill_vllm
    echo "Launching vLLM server on GPU 1..."
    CUDA_VISIBLE_DEVICES=1 VLLM_SERVER_DEV_MODE=1 nohup vllm serve "${MODEL}" \
        --dtype bfloat16 \
        --max-model-len 768 \
        --gpu-memory-utilization 0.90 \
        --logprobs-mode processed_logprobs \
        --weight-transfer-config '{"backend":"nccl"}' \
        --port "${VLLM_PORT}" \
        > "${RESULTS_DIR}/vllm_server.log" 2>&1 &
    wait_for_vllm
}

# --- Main ---
mkdir -p "${RESULTS_DIR}"

for run in $(seq 1 ${NUM_REPEATS}); do
    echo "========== REPEAT ${run}/${NUM_REPEATS} =========="

    # Run 1: GRPO baseline
    launch_vllm
    echo "[Run ${run}] Starting GRPOTrainer baseline..."
    CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/grpo_baseline.py" \
        --run_id "repeat_${run}" 2>&1 | tee "${RESULTS_DIR}/grpo_baseline/run_${run}.log"
    kill_vllm

    # Run 2: AsyncGRPO (default)
    launch_vllm
    echo "[Run ${run}] Starting AsyncGRPOTrainer (default)..."
    CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/async_grpo_main.py" \
        --run_id "repeat_${run}" 2>&1 | tee "${RESULTS_DIR}/async_grpo_main/run_${run}.log"
    kill_vllm

    # Run 3: AsyncGRPO (tuned)
    launch_vllm
    echo "[Run ${run}] Starting AsyncGRPOTrainer (tuned)..."
    CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/async_grpo_tuned.py" \
        --run_id "repeat_${run}" 2>&1 | tee "${RESULTS_DIR}/async_grpo_tuned/run_${run}.log"
    kill_vllm
done

echo "All runs complete. Results in ${RESULTS_DIR}/"
```

---

## Metrics Collection

### Throughput / Latency

| Metric | Source | Notes |
|--------|--------|-------|
| Wall-clock time | `time.time()` around `trainer.train()` | Total training time |
| Step time (s) | Trainer logs (`train/step_time`) | Per-step average |
| Iterations/sec | `max_steps / wall_time` | Excludes warmup (first 20 steps) |
| Tokens/sec | `total_generated_tokens / wall_time` | From completion lengths × num_generations |

### Correctness

| Metric | Source | Expectation |
|--------|--------|-------------|
| No crashes/NaN/Inf | Training logs | Must pass |
| `reward/accuracy_reward/mean` | Trainer metrics | Async tracks GRPO trend |
| `kl` | Trainer metrics | Same regime (no collapse/explosion) |
| `entropy` | Trainer metrics | Same regime |
| `clip_ratio` | Trainer metrics | Comparable |
| `completions/mean_length` | Trainer metrics | Comparable |
| `completions/clipped_ratio` | Trainer metrics | Comparable |

### Async-Specific Diagnostics

| Metric | Source | Healthy range |
|--------|--------|---------------|
| Stale-sample drop rate | Queue dataset logs | < 30% |
| `weight_sync_time_s` | Trainer metrics | < 5s per sync |
| `queue_wait_time_s` | Trainer metrics | < 10s sustained |
| `generation_tok_per_s` | vLLM server stats | Stable, not degrading |

---

## Success Criteria

### 1. Correctness Parity (trend-level)

- [ ] Async run is stable: no NaN, no crash, no hang/deadlock
- [ ] Reward trend follows GRPO baseline direction
- [ ] Final-window (last 50 steps) reward mean within ~10% relative gap of GRPO
- [ ] KL, entropy, clip_ratio remain in same regime (no collapse/explosion)
- [ ] **Optional**: held-out eval (200 GSM8K test prompts) accuracy gap ≤ 3 absolute points

### 2. Throughput Gain

- [ ] AsyncGRPO (default) shows measurable speedup over GRPO baseline
- [ ] Expected range: **1.1x – 1.6x** on 2× A10G with 4B model
- [ ] Tuned variant shows equal or better throughput than default

### 3. Gap-Closure Coverage Confirmed

- [ ] Logs show model loaded via `create_model_from_path` (string model arg)
- [ ] `model_init_kwargs` applied (bfloat16 + flash_attention_2 visible in config dump)
- [ ] `AutoProcessor.from_pretrained` called (no explicit `processing_class` passed)
- [ ] Non-default sampling params visible in vLLM request logs (`top_p=0.95`, `top_k=50`, `min_p=0.05`, `repetition_penalty=1.1`)
- [ ] PEFT/LoRA active: merge/unmerge cycle visible in weight sync logs
- [ ] Model is `Qwen/Qwen3-4B` (vLLM 0.18 compatible, exercises all code paths)

---

## Memory Budget Estimation (2× A10G, 24 GB each)

### GPU 0 (Trainer)

| Component | Estimate |
|-----------|----------|
| Qwen3-4B (bf16) | ~8 GB |
| LoRA adapters (r=16) | ~0.2 GB |
| Optimizer states (AdamW on LoRA params) | ~0.4 GB |
| Activations (gradient checkpointing, bs=4) | ~4-6 GB |
| **Total** | **~13-15 GB** ✓ |

### GPU 1 (vLLM server)

| Component | Estimate |
|-----------|----------|
| Qwen3-4B (bf16) | ~8 GB |
| KV cache (max_model_len=768, 8 seqs) | ~4-6 GB |
| vLLM overhead | ~2 GB |
| **Total** | **~14-16 GB** ✓ |

> Both fit comfortably within 24 GB. If memory pressure appears, reduce `num_generations` to 4 or `max_completion_length` to 256.

---

## Fallback: Memory-Pressure Adjustments

If OOM occurs:

1. Reduce `num_generations`: 8 → 4
2. Reduce `max_completion_length`: 384 → 256
3. Reduce `per_device_train_batch_size`: 4 → 2 (increase `gradient_accumulation_steps` to 4)
4. Reduce `--gpu-memory-utilization`: 0.90 → 0.85
5. Last resort: switch to `Qwen/Qwen3-1.7B` (still validates all code paths)

---

## Optional: Held-Out Evaluation Script

After training completes, evaluate checkpoints on GSM8K test split:

```python
# eval_checkpoint.py
# Load checkpoint, generate on 200 GSM8K test prompts with greedy decoding,
# compute exact-match accuracy against ground truth.
# Compare GRPO vs Async checkpoints at same step count.
```

---

## Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Environment setup (deps, model download) | ~30 min | One-time |
| vLLM server validation | ~5 min | Health check + single generation test |
| Run 1 (GRPO baseline, 300 steps) | ~30-45 min | Depends on generation throughput |
| Run 2 (AsyncGRPO default, 300 steps) | ~20-35 min | Expected faster |
| Run 3 (AsyncGRPO tuned, 300 steps) | ~20-35 min | Expected similar to Run 2 |
| Repeats (×3 total) | ~3-5 hours | Full benchmark suite |
| Analysis & report | ~30 min | Compare metrics |
| **Total** | **~4-6 hours** | |

---

## Execution Checklist

- [ ] Confirm 2× A10G available (`nvidia-smi`)
- [ ] Install vLLM (accepts torch 2.12→2.11 downgrade, keeps transformers 5.9):
  ```bash
  uv pip install --python /root/trl/.venv/bin/python "vllm==0.21.0" "transformers>=5.9"
  ```
  > Note: TRL pins `vllm<=0.18.0` but 0.18 requires torch 2.10 (ABI-incompatible with our
  > torch 2.12 build). vLLM 0.21 requires torch 2.11 which works. The server-mode HTTP API
  > is stable across versions. TRL will emit a warning but functions correctly.
- [ ] Install remaining deps: `uv pip install --python /root/trl/.venv/bin/python math-verify latex2sympy2_extended`
- [ ] Download model: `huggingface-cli download Qwen/Qwen3-4B`
- [ ] Download dataset: `python -c "from datasets import load_dataset; load_dataset('openai/gsm8k', 'main')"`
- [ ] Validate vLLM server starts and serves completions
- [ ] Run benchmark suite (`run_benchmark.sh`)
- [ ] Collect results and generate comparison report
- [ ] Verify all gap-closure checkboxes are satisfied
