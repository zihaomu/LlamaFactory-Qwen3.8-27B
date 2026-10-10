#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if ! command -v "$PYTHON_BIN" > /dev/null; then
  printf 'Python 3.12 is required; %s was not found\n' "$PYTHON_BIN" >&2
  exit 2
fi

python_version="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$python_version" != "3.12" ]]; then
  printf 'Python 3.12 is required for the pinned ROCm wheels, got %s\n' "$python_version" >&2
  exit 2
fi

if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$PROJECT_ROOT/.venv"
fi

source "$SCRIPT_DIR/env.sh"

python -m pip install --upgrade pip packaging wheel setuptools editables 'hatchling>=1.18.0'
python -m pip install -r "$PROJECT_ROOT/requirements/rocm-7.2.4-py312.txt"

if [[ ! -d "$PROJECT_ROOT/src/LLaMA-Factory/.git" ]]; then
  git clone --filter=blob:none https://github.com/hiyouga/LLaMA-Factory.git \
    "$PROJECT_ROOT/src/LLaMA-Factory"
fi

git -C "$PROJECT_ROOT/src/LLaMA-Factory" fetch --quiet origin "$LLAMAFACTORY_REVISION"
git -C "$PROJECT_ROOT/src/LLaMA-Factory" checkout --detach "$LLAMAFACTORY_REVISION"

python -m pip install \
  -r "$PROJECT_ROOT/requirements/training-pins.txt" \
  -e "$PROJECT_ROOT/src/LLaMA-Factory"
python -m pip check
python "$PROJECT_ROOT/scripts/validate_repository.py"

printf 'SETUP_PASS python=%s llamafactory_revision=%s\n' \
  "$(python --version 2>&1)" \
  "$(git -C "$PROJECT_ROOT/src/LLaMA-Factory" rev-parse HEAD)"