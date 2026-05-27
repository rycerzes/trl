#!/usr/bin/env bash
# AsyncGRPO vs GRPO benchmark orchestrator
# - GRPO baseline: colocate mode (no separate server needed)
# - AsyncGRPO: server mode (vLLM launched on GPU 1, trainer on GPU 0)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../results"
VLLM_PORT=8000
MODEL="Qwen/Qwen3-4B"
VENV="/root/trl/.venv/bin"
NUM_REPEATS="${NUM_REPEATS:-1}"

# --- Helpers ---
wait_for_vllm() {
    echo "[vllm] Waiting for server on port ${VLLM_PORT}..."
    for i in $(seq 1 120); do
        if curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo "[vllm] Server ready (took $((i * 2))s)"
            return 0
        fi
        sleep 2
    done
    echo "[vllm] ERROR: Server did not start within 240s"
    echo "[vllm] Last 20 lines of server log:"
    tail -20 "${RESULTS_DIR}/vllm_server.log" 2>/dev/null || true
    return 1
}

kill_vllm() {
    if pgrep -f "vllm.entrypoints" > /dev/null 2>&1; then
        echo "[vllm] Stopping server..."
        pkill -f "vllm.entrypoints" 2>/dev/null || true
        sleep 5
    fi
}

launch_vllm() {
    kill_vllm
    # NOTE: Do NOT use CUDA_VISIBLE_DEVICES to isolate GPUs for NCCL weight transfer.
    # Both processes must see all GPUs so NCCL can resolve the physical topology.
    # vLLM defaults to GPU 0; the trainer uses cuda:1 via device index.
    # NCCL_P2P_DISABLE=1 + NCCL_SHM_DISABLE=1 required on PCIe-only hardware (no NVLink).
    # --gpu-memory-utilization 0.45 leaves room for NCCL weight transfer buffers (~1GB).
    echo "[vllm] Launching server on GPU 0 (model: ${MODEL})..."
    NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 VLLM_SERVER_DEV_MODE=1 \
        "${VENV}/python" -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --dtype bfloat16 \
        --max-model-len 768 \
        --gpu-memory-utilization 0.45 \
        --logprobs-mode processed_logprobs \
        --weight-transfer-config '{"backend":"nccl"}' \
        --port "${VLLM_PORT}" \
        --enforce-eager \
        > "${RESULTS_DIR}/vllm_server.log" 2>&1 &
    wait_for_vllm
}

# Trap to ensure cleanup on exit
trap kill_vllm EXIT

# --- Pre-flight checks ---
echo "============================================"
echo " AsyncGRPO vs GRPO Benchmark"
echo "============================================"
echo ""
echo "Environment:"
"${VENV}/python" -c "
import torch, vllm, transformers, peft, trl
print(f'  torch:        {torch.__version__}')
print(f'  vllm:         {vllm.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  peft:         {peft.__version__}')
print(f'  trl:          {trl.__version__}')
print(f'  GPUs:         {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}')
" 2>/dev/null
echo ""
echo "Config:"
echo "  Model:       ${MODEL}"
echo "  Repeats:     ${NUM_REPEATS}"
echo "  GRPO:        colocate mode (both GPUs)"
echo "  AsyncGRPO:   server mode (GPU 0 trainer, GPU 1 vLLM)"
echo ""

# --- Create output dirs ---
mkdir -p "${RESULTS_DIR}/grpo_baseline"
mkdir -p "${RESULTS_DIR}/async_grpo_main"

# --- Main loop ---
for run in $(seq 1 "${NUM_REPEATS}"); do
    echo ""
    echo "========== REPEAT ${run}/${NUM_REPEATS} =========="
    echo ""

    # --- Run 1: GRPO baseline (colocate, no separate server) ---
    echo "[bench] Starting GRPOTrainer baseline - colocate mode (run ${run})..."
    CUDA_VISIBLE_DEVICES=0,1 "${VENV}/python" "${SCRIPT_DIR}/grpo_baseline.py" \
        2>&1 | tee "${RESULTS_DIR}/grpo_baseline/run_${run}.log"
    echo ""

    # --- Run 2: AsyncGRPO (server mode) ---
    # NOTE: vLLM launched without CUDA_VISIBLE_DEVICES (sees all GPUs, uses GPU 0).
    # Trainer uses CUDA_VISIBLE_DEVICES=1 so HF Accelerator places model on GPU 1.
    # NCCL resolves physical topology by bus ID across processes.
    # NCCL_P2P_DISABLE + NCCL_SHM_DISABLE needed on both sides for PCIe-only hardware.
    launch_vllm
    echo "[bench] Starting AsyncGRPOTrainer - server mode (run ${run})..."
    NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 CUDA_VISIBLE_DEVICES=1 "${VENV}/python" "${SCRIPT_DIR}/async_grpo_main.py" \
        2>&1 | tee "${RESULTS_DIR}/async_grpo_main/run_${run}.log"
    kill_vllm
    echo ""
done

# --- Summary ---
echo ""
echo "============================================"
echo " Benchmark Complete"
echo "============================================"
echo ""
echo "Results:"
for variant in grpo_baseline async_grpo_main; do
    echo "  ${variant}:"
    for f in "${RESULTS_DIR}/${variant}"/timing.json; do
        if [ -f "$f" ]; then
            "${VENV}/python" -c "
import json
d = json.load(open('$f'))
print(f'    wall_time: {d[\"wall_time_s\"]:.1f}s  steps/s: {d[\"steps_per_sec\"]:.3f}')
" 2>/dev/null
        fi
    done
done
echo ""
echo "Full logs in: ${RESULTS_DIR}/"
