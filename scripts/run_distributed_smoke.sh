#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh"

GPU_IDS="${GPU_IDS:-0,1}"
IFS=',' read -r -a selected_gpus <<< "$GPU_IDS"
if [[ "${#selected_gpus[@]}" -ne 2 || "${selected_gpus[0]}" == "${selected_gpus[1]}" ]]; then
  printf 'GPU_IDS must contain two distinct comma-separated IDs\n' >&2
  exit 2
fi
for gpu_id in "${selected_gpus[@]}"; do
  if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
    printf 'GPU IDs must be numeric, got %s\n' "$gpu_id" >&2
    exit 2
  fi
done

export HIP_VISIBLE_DEVICES="$GPU_IDS"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  "$SCRIPT_DIR/check_distributed.py"