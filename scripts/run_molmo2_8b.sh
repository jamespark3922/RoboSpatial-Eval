#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-config.yaml}"
MODEL_PATH="${MODEL_PATH:-allenai/Molmo2-8B}"

python main.py molmo2 "${MODEL_PATH}" --config "${CONFIG}" "$@"
