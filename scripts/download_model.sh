#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh"

if ! command -v hf > /dev/null; then
  printf 'The hf CLI is missing. Run scripts/setup.sh first.\n' >&2
  exit 2
fi

mkdir -p "$MODEL_PATH"
hf download "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --local-dir "$MODEL_PATH" \
  --max-workers "${HF_DOWNLOAD_WORKERS:-4}"

hf download "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --local-dir "$MODEL_PATH" \
  --dry-run

if [[ ! -f "$MODEL_PATH/config.json" || ! -f "$MODEL_PATH/model.safetensors.index.json" ]]; then
  printf 'Model download is incomplete: %s\n' "$MODEL_PATH" >&2
  exit 1
fi

printf 'MODEL_DOWNLOAD_PASS id=%s revision=%s path=%s\n' \
  "$MODEL_ID" "$MODEL_REVISION" "$MODEL_PATH"