#!/usr/bin/env bash
# AsyncGRPO vs GRPO benchmark orchestrator
# Both configurations use 2 GPUs:
# - GRPO baseline: server mode (TRL vllm-serve on GPU 1, trainer on GPU 0)
# - AsyncGRPO:     server mode (vLLM on GPU 0, trainer on GPU 1) with NCCL weight sync
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../results"
VLLM_PORT=8000
MODEL="Qwen/Qwen3-4B"
VENV="/root/trl/.venv/bin"
NUM_REPEATS="${NUM_REPEATS:-1}"

# --- Helpers ---
SERVER_PGID=""

wait_for_server() {
    local health_url="$1"
    local timeout="${2:-240}"
    echo "[server] Waiting for server at ${health_url}..."
    for i in $(seq 1 $((timeout / 2))); do
        if curl -sf "${health_url}" > /dev/null 2>&1; then
            echo "[server] Ready (took $((i * 2))s)"
            return 0
        fi
        sleep 2
    done
    echo "[server] ERROR: Server did not start within ${timeout}s"
    return 1
}

kill_server() {
    # Kill the server's entire process group to avoid orphaning CUDA worker children.
    # Orphaned CUDA workers leak GPU memory to PID 1 permanently in containers.
    if [ -n "${SERVER_PGID}" ]; then
        echo "[server] Stopping process group ${SERVER_PGID}..."
        kill -- -"${SERVER_PGID}" 2>/dev/null || true
        SERVER_PGID=""
    fi
    # Fallback: catch anything still lingering (SIGTERM only)
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    pkill -f "trl vllm-serve" 2>/dev/null || true
    pkill -f "vllm_serve" 2>/dev/null || true
    sleep 5  # allow CUDA context cleanup before next launch
}

launch_trl_vllm_serve() {
    # TRL's own vllm-serve provides /generate/, /init_communicator/, /update_named_param/
    # Used by the GRPO baseline in server mode
    kill_server
    echo "[server] Launching TRL vllm-serve on GPU 1 (model: ${MODEL})..."
    setsid env CUDA_VISIBLE_DEVICES=1 "${VENV}/trl" vllm-serve \
        --model "${MODEL}" \
        --dtype bfloat16 \
        --max_model_len 768 \
        --gpu_memory_utilization 0.90 \
        --enforce_eager \
        --port "${VLLM_PORT}" \
        > "${RESULTS_DIR}/trl_vllm_serve.log" 2>&1 &
    SERVER_PGID=$(ps -o pgid= -p $! | tr -d ' ')
    wait_for_server "http://localhost:${VLLM_PORT}/health/"
}

launch_vllm_server() {
    # Standard vLLM serve with RLHF weight transfer for AsyncGRPO
    # NOTE: Do NOT use CUDA_VISIBLE_DEVICES — vLLM must see all GPUs for NCCL topology.
    # vLLM defaults to GPU 0 with TP=1.
    kill_server
    echo "[server] Launching vLLM server on GPU 0 (model: ${MODEL})..."
    setsid env NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 VLLM_SERVER_DEV_MODE=1 \
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
    SERVER_PGID=$(ps -o pgid= -p $! | tr -d ' ')
    wait_for_server "http://localhost:${VLLM_PORT}/health"
}

# Trap to ensure cleanup on exit
trap kill_server EXIT

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
echo "  GRPO:        server mode (GPU 0 trainer, GPU 1 TRL vllm-serve)"
echo "  AsyncGRPO:   server mode (GPU 1 trainer, GPU 0 vLLM + NCCL weight sync)"
echo ""

# --- Create output dirs ---
mkdir -p "${RESULTS_DIR}/grpo_baseline"
mkdir -p "${RESULTS_DIR}/async_grpo_main"

# --- Main loop ---
for run in $(seq 1 "${NUM_REPEATS}"); do
    echo ""
    echo "========== REPEAT ${run}/${NUM_REPEATS} =========="
    echo ""

    # --- Run 1: GRPO baseline (server mode, TRL vllm-serve on GPU 1) ---
    launch_trl_vllm_serve
    echo "[bench] Starting GRPOTrainer baseline - server mode (run ${run})..."
    CUDA_VISIBLE_DEVICES=0 "${VENV}/python" "${SCRIPT_DIR}/grpo_baseline.py" \
        2>&1 | tee "${RESULTS_DIR}/grpo_baseline/run_${run}.log"
    kill_server
    echo ""

    # --- Run 2: AsyncGRPO (server mode, vLLM on GPU 0, trainer on GPU 1) ---
    launch_vllm_server
    echo "[bench] Starting AsyncGRPOTrainer - server mode (run ${run})..."
    NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 CUDA_VISIBLE_DEVICES=1 \
        "${VENV}/python" "${SCRIPT_DIR}/async_grpo_main.py" \
        2>&1 | tee "${RESULTS_DIR}/async_grpo_main/run_${run}.log"
    kill_server
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
