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
GRPOTrainer baseline benchmark (sync, vLLM colocate mode).

No separate vLLM server needed — vLLM runs colocated on the same GPU.

Run:
    CUDA_VISIBLE_DEVICES=0,1 python benchmarks/scripts/grpo_baseline.py

Note: GRPOTrainer's server-mode weight sync uses /init_communicator/ which was removed in
vLLM 0.21. Colocate mode avoids this API incompatibility while still using vLLM for
generation. The AsyncGRPOTrainer uses the newer /init_weight_transfer_engine endpoint
which works with vLLM 0.21.
"""

import json
import time
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig

from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward


MODEL_ID = "Qwen/Qwen3-4B"
OUTPUT_DIR = "benchmarks/results/grpo_baseline"


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

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        # vLLM colocate mode (avoids 0.21 /init_communicator API break)
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.45,
        vllm_max_model_length=768,
        vllm_enable_sleep_mode=True,
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

    trainer = GRPOTrainer(
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
            "peft_r": 16,
            "variant": "grpo_baseline",
        },
    }
    results_path = Path(OUTPUT_DIR) / "timing.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\n{'='*60}")
    print(f"GRPO Baseline complete: {wall_time:.1f}s ({config.max_steps / wall_time:.2f} steps/s)")
    print(f"Results saved to {results_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
