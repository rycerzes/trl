# AsyncGRPO: make NCCL weight sync robust on PCIe-only GPUs (auto fallback from P2P/SHM)

## Feature request

Please add a topology-aware NCCL fallback for AsyncGRPO weight sync:

- Detect when multi-GPU machines have no usable peer access / NVLink.
- Automatically set:
  - `NCCL_P2P_DISABLE=1`
  - `NCCL_SHM_DISABLE=1`
- Respect explicit user overrides if env vars are already set.
- Log a clear warning with override instructions.

Additionally, document that for `vllm_mode="server"` / remote vLLM, these env vars may need to be set on the vLLM server process too.

---

## Motivation

On PCIe-only or constrained topologies (e.g. many cloud A10/L4/T4 setups), NCCL P2P/SHM can hang during large broadcast/allreduce operations used in trainer→inference weight sync.

Other RL frameworks already treat this as an operational reliability issue:

### Frameworks that proactively handle this

1. **Prime-RL**
   - Topology-aware helper:
     - https://github.com/PrimeIntellect-ai/prime-rl/blob/10bc02c6dccf34f7e1f86af83794de92d0dbb60c/src/prime_rl/utils/nccl.py
   - Called from both inference and trainer NCCL paths:
     - https://github.com/PrimeIntellect-ai/prime-rl/blob/10bc02c6dccf34f7e1f86af83794de92d0dbb60c/src/prime_rl/inference/vllm/worker/nccl.py
     - https://github.com/PrimeIntellect-ai/prime-rl/blob/10bc02c6dccf34f7e1f86af83794de92d0dbb60c/src/prime_rl/trainer/rl/broadcast/nccl.py

2. **SkyRL**
   - Peer-access probe and runtime env fallback (`NCCL_P2P_DISABLE`/`NCCL_SHM_DISABLE`):
     - https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl/train/utils/utils.py

### Frameworks that mostly rely on other NCCL knobs

- NeMo-RL / VERL / AReaL commonly apply `NCCL_CUMEM_ENABLE` / `NCCL_NVLS_ENABLE` workarounds, but do not generally auto-apply P2P+SHM disable fallback for this case.

Given AsyncGRPO’s frequent NCCL weight sync operations, this fallback significantly improves out-of-the-box stability on non-NVLink hardware.

---

## Your contribution

I can open a PR for this, poc in 