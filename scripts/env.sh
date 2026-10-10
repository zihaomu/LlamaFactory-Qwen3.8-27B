#!/usr/bin/env bash

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PROJECT_ROOT

export VIRTUAL_ENV="$PROJECT_ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$PROJECT_ROOT/cache/modelscope}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PROJECT_ROOT/cache/pip}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PROJECT_ROOT/cache/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/cache/xdg}"
export TMPDIR="${TMPDIR:-$PROJECT_ROOT/cache/tmp}"

export PIP_DISABLE_PIP_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false

source "$PROJECT_ROOT/versions.env"

export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3.8-27B}"
export DATASET_DIR="${DATASET_DIR:-$PROJECT_ROOT/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"

mkdir -p \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$MODELSCOPE_CACHE" \
  "$PIP_CACHE_DIR" \
  "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$TMPDIR" \
  "$OUTPUT_ROOT"