#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-config.yaml}"
MODEL_PATH="${MODEL_PATH:-allenai/MolmoPoint-8B}"

python main.py molmopoint "${MODEL_PATH}" --config "${CONFIG}" "$@"
