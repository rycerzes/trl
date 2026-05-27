# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AsyncGRPOTrainer benchmark (default async knobs).

Launch vLLM server first (no CUDA_VISIBLE_DEVICES — sees all GPUs, uses GPU 0 by default):
    VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-4B \
        --dtype bfloat16 --max-model-len 768 --gpu-memory-utilization 0.90 \
        --logprobs-mode processed_logprobs \
        --weight-transfer-config '{"backend":"nccl"}' --port 8000

Then run (trainer on GPU 1):
    CUDA_VISIBLE_DEVICES=1 python benchmarks/scripts/async_grpo_main.py
"""

import json
import time
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.rewards import accuracy_reward


MODEL_ID = "Qwen/Qwen3-4B"
OUTPUT_DIR = "benchmarks/results/async_grpo_main"


def format_sample(sample):
    return {
        "prompt": [{"role": "user", "content": sample["question"]}],
        "solution": sample["answer"].split("####")[-1].strip(),
    }


def main():
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    config = AsyncGRPOConfig(
        output_dir=OUTPUT_DIR,
        # vLLM server
        vllm_server_base_url="http://localhost:8000",
        # Async defaults
        max_staleness=4,
        weight_sync_steps=1,
        max_inflight_tasks=-1,  # auto
        # Generation (non-default sampling to exercise plumbing)
        max_completion_length=384,
        num_generations=8,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        min_p=0.05,
        repetition_penalty=1.1,
        # Training
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        max_steps=300,
        learning_rate=1e-5,
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        bf16=True,
        # Saving & logging
        save_strategy="steps",
        save_steps=100,
        logging_steps=5,
        log_completions=True,
        report_to="none",
        # Chat template
        chat_template_kwargs={"enable_thinking": False},
    )

    trainer = AsyncGRPOTrainer(
        model=MODEL_ID,
        args=config,
        train_dataset=dataset,
        reward_funcs=accuracy_reward,
        peft_config=peft_config,
        # processing_class=None → triggers AutoProcessor.from_pretrained
    )

    # Run training with timing
    start_time = time.time()
    trainer.train()
    wall_time = time.time() - start_time

    # Save timing results
    results = {
        "wall_time_s": wall_time,
        "max_steps": config.max_steps,
        "steps_per_sec": config.max_steps / wall_time,
        "config": {
            "model": MODEL_ID,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_completion_length": config.max_completion_length,
            "num_generations": config.num_generations,
            "max_staleness": config.max_staleness,
            "weight_sync_steps": config.weight_sync_steps,
            "max_inflight_tasks": config.max_inflight_tasks,
            "peft_r": 16,
            "variant": "async_grpo_main",
        },
    }
    results_path = Path(OUTPUT_DIR) / "timing.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\n{'='*60}")
    print(f"AsyncGRPO (default) complete: {wall_time:.1f}s ({config.max_steps / wall_time:.2f} steps/s)")
    print(f"Results saved to {results_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
