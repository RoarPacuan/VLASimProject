#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GR00T_ROOT="${GR00T_ROOT:-$ROOT_DIR/submodules/Isaac-GR00T}"
BASE_MODEL_PATH="nvidia/GR00T-N1.6-3B"
DATASET_PATH=""
OUTPUT_DIR=""
EMBODIMENT_TAG="NEW_EMBODIMENT"
MODALITY_CONFIG_PATH=""
NUM_GPUS="1"
GLOBAL_BATCH_SIZE="16"
MAX_STEPS="30000"
EXTRA_ARGS=()

usage() {
    cat <<EOF
Fine-tune GR00T N1.6 using Arena-compatible LeRobot data.

Required: --dataset-path, --output-dir
Options:
  --base-model-path <path>   Hugging Face model or local checkpoint.
  --dataset-path <path>      GR00T-flavored LeRobot v2 dataset.
  --output-dir <path>        Checkpoint output directory.
  --embodiment-tag <tag>     NEW_EMBODIMENT, GR1, or OXE_DROID.
  --modality-config-path <p> Python modality config for NEW_EMBODIMENT.
  --num-gpus <n>              Number of GPUs (default: 1).
  --global-batch-size <n>     Effective batch size (default: 16).
  --max-steps <n>             Training steps (default: 30000).
  -- <args>                   Extra launch_finetune.py arguments.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-model-path) BASE_MODEL_PATH="$2"; shift 2 ;;
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --embodiment-tag) EMBODIMENT_TAG="$2"; shift 2 ;;
        --modality-config-path) MODALITY_CONFIG_PATH="$2"; shift 2 ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --global-batch-size) GLOBAL_BATCH_SIZE="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$DATASET_PATH" && -n "$OUTPUT_DIR" ]] || { usage >&2; exit 2; }
[[ -d "$GR00T_ROOT" ]] || { echo "GR00T checkout not found: $GR00T_ROOT" >&2; exit 1; }
[[ -d "$DATASET_PATH" ]] || { echo "Dataset not found: $DATASET_PATH" >&2; exit 1; }

TRAIN_ARGS=(
    --base-model-path "$BASE_MODEL_PATH"
    --dataset-path "$DATASET_PATH"
    --output-dir "$OUTPUT_DIR"
    --embodiment-tag "$EMBODIMENT_TAG"
    --num-gpus "$NUM_GPUS"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --max-steps "$MAX_STEPS"
    --no-tune-llm --tune-visual --tune-projector --tune-diffusion-model)
if [[ -n "$MODALITY_CONFIG_PATH" ]]; then
    TRAIN_ARGS+=(--modality-config-path "$MODALITY_CONFIG_PATH")
fi
TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

cd "$GR00T_ROOT"
if [[ "$NUM_GPUS" -gt 1 ]]; then
    uv run python -m torch.distributed.run \
        --nproc_per_node="$NUM_GPUS" --standalone \
        gr00t/experiment/launch_finetune.py "${TRAIN_ARGS[@]}"
else
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run python \
        gr00t/experiment/launch_finetune.py "${TRAIN_ARGS[@]}"
fi