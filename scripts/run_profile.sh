#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh"
cd "$PROJECT_ROOT"

PROFILE="${1:-}"
case "$PROFILE" in
  bf16-r64)
    CONFIG_PATH="$PROJECT_ROOT/configs/recommended/bf16-lora-r64-2x-w7900.yaml"
    GPU_COUNT=2
    DEFAULT_GPU_IDS="0,1"
    ;;
  nf4-r128)
    CONFIG_PATH="$PROJECT_ROOT/configs/recommended/nf4-qlora-r128-1x-w7900.yaml"
    GPU_COUNT=1
    DEFAULT_GPU_IDS="0"
    ;;
  *)
    printf 'Usage: %s {bf16-r64|nf4-r128}\n' "$0" >&2
    exit 2
    ;;
esac

DRY_RUN="${DRY_RUN:-0}"
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  printf 'DRY_RUN must be 0 or 1\n' >&2
  exit 2
fi
if [[ "$DRY_RUN" == "0" ]]; then
  if ! command -v llamafactory-cli > /dev/null; then
    printf 'llamafactory-cli is missing. Run scripts/setup.sh first.\n' >&2
    exit 2
  fi
  if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    printf 'Model config is missing at %s. Run scripts/download_model.sh first.\n' "$MODEL_PATH" >&2
    exit 2
  fi
fi

GPU_IDS="${GPU_IDS:-$DEFAULT_GPU_IDS}"
IFS=',' read -r -a selected_gpus <<< "$GPU_IDS"
if [[ "${#selected_gpus[@]}" -ne "$GPU_COUNT" ]]; then
  printf '%s requires %s GPU ID(s), got %s\n' "$PROFILE" "$GPU_COUNT" "$GPU_IDS" >&2
  exit 2
fi
declare -A seen_gpus=()
for gpu_id in "${selected_gpus[@]}"; do
  if [[ ! "$gpu_id" =~ ^[0-9]+$ || -n "${seen_gpus[$gpu_id]:-}" ]]; then
    printf 'GPU_IDS must contain distinct numeric IDs, got %s\n' "$GPU_IDS" >&2
    exit 2
  fi
  seen_gpus[$gpu_id]=1
done

DATASET="${DATASET:-example_sft}"
MAX_STEPS="${MAX_STEPS:-100}"
CUTOFF_LEN="${CUTOFF_LEN:-512}"
if [[ ! "$MAX_STEPS" =~ ^[1-9][0-9]*$ || ! "$CUTOFF_LEN" =~ ^[1-9][0-9]*$ ]]; then
  printf 'MAX_STEPS and CUTOFF_LEN must be positive integers\n' >&2
  exit 2
fi
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$OUTPUT_ROOT/$PROFILE/$RUN_ID"
ADAPTER_DIR="$RUN_DIR/adapter"

train_args=(
  "$CONFIG_PATH"
  "model_name_or_path=$MODEL_PATH"
  "dataset=$DATASET"
  "dataset_dir=$DATASET_DIR"
  "cutoff_len=$CUTOFF_LEN"
  "max_steps=$MAX_STEPS"
  "output_dir=$ADAPTER_DIR"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'profile=%s\n' "$PROFILE"
  printf 'gpu_count=%s\n' "$GPU_COUNT"
  printf 'gpu_ids=%s\n' "$GPU_IDS"
  printf 'dataset=%s\n' "$DATASET"
  printf 'max_steps=%s\n' "$MAX_STEPS"
  printf 'cutoff_len=%s\n' "$CUTOFF_LEN"
  printf 'command=llamafactory-cli train'
  printf ' %q' "${train_args[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$RUN_DIR"
cp "$CONFIG_PATH" "$RUN_DIR/config.yaml"

export HIP_VISIBLE_DEVICES="$GPU_IDS"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
if [[ "$GPU_COUNT" -eq 2 ]]; then
  export FORCE_TORCHRUN=1
  export NPROC_PER_NODE=2
else
  unset FORCE_TORCHRUN NPROC_PER_NODE LOCAL_RANK RANK WORLD_SIZE
fi

{
  printf 'HIP_VISIBLE_DEVICES=%q ' "$HIP_VISIBLE_DEVICES"
  if [[ "$GPU_COUNT" -eq 2 ]]; then
    printf 'FORCE_TORCHRUN=1 NPROC_PER_NODE=2 '
  fi
  printf 'llamafactory-cli train'
  printf ' %q' "${train_args[@]}"
  printf '\n'
} > "$RUN_DIR/command.txt"

{
  printf 'profile=%s\n' "$PROFILE"
  printf 'physical_gpu_ids=%s\n' "$GPU_IDS"
  printf 'model_id=%s\n' "$MODEL_ID"
  printf 'model_revision=%s\n' "$MODEL_REVISION"
  printf 'llamafactory_revision=%s\n' "$(git -C "$PROJECT_ROOT/src/LLaMA-Factory" rev-parse HEAD)"
  printf 'python=%s\n' "$(python --version 2>&1)"
  printf 'kernel=%s\n' "$(uname -r)"
  python - <<'PY'
from importlib.metadata import version

for package in ("torch", "transformers", "deepspeed", "bitsandbytes", "peft"):
    print(f"{package}={version(package)}")
PY
} > "$RUN_DIR/env.txt" 2>&1

monitor_pid=""
stop_monitor() {
  if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
}
trap stop_monitor EXIT INT TERM

if command -v amd-smi > /dev/null; then
  amd-smi metric --mem-usage --gpu "${selected_gpus[@]}" > "$RUN_DIR/gpu_snapshot_before.txt"
  amd-smi monitor \
    --power-usage \
    --temperature \
    --gfx \
    --mem \
    --vram-usage \
    --watch 1 \
    --gpu "${selected_gpus[@]}" \
    --csv \
    --file "$RUN_DIR/gpu_monitor.csv" \
    > "$RUN_DIR/gpu_monitor.log" 2>&1 &
  monitor_pid="$!"
fi

set +e
llamafactory-cli train "${train_args[@]}" 2>&1 | tee "$RUN_DIR/train.log"
run_status="${PIPESTATUS[0]}"
set -e

stop_monitor
trap - EXIT INT TERM
if command -v amd-smi > /dev/null; then
  amd-smi metric --mem-usage --gpu "${selected_gpus[@]}" > "$RUN_DIR/gpu_snapshot_after.txt"
fi
printf '%s\n' "$run_status" > "$RUN_DIR/exit_code.txt"

for metrics_file in all_results.json train_results.json eval_results.json; do
  if [[ -f "$ADAPTER_DIR/$metrics_file" ]]; then
    cp "$ADAPTER_DIR/$metrics_file" "$RUN_DIR/$metrics_file"
  fi
done

if [[ "$run_status" -ne 0 || ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]]; then
  final_status="$run_status"
  if [[ "$final_status" -eq 0 ]]; then
    final_status=1
  fi
  printf 'PROFILE_FAIL profile=%s run_dir=%s exit_code=%s\n' \
    "$PROFILE" "$RUN_DIR" "$final_status" >&2
  exit "$final_status"
fi

printf 'PROFILE_PASS profile=%s run_dir=%s\n' "$PROFILE" "$RUN_DIR"