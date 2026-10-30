#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-}"
CONFIG=""
EXP_NAME="arena_pi0_finetune"
EXTRA_ARGS=()

usage() {
    cat <<EOF
Run OpenPI fine-tuning for a Pi0 checkpoint.

Arena supplies Pi0 inference and simulation adapters, but training is owned by OpenPI.
This wrapper keeps the OpenPI checkout and command explicit because its training configs
change independently of Arena.

Usage:
  OPENPI_ROOT=/path/to/openpi $(basename "$0") --config <openpi-config> [--exp-name <name>] -- [train args]

The arguments after -- are passed to OpenPI's scripts/train.py.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --exp-name) EXP_NAME="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ -n "$OPENPI_ROOT" ]] || { echo "Set OPENPI_ROOT to an OpenPI checkout." >&2; exit 2; }
[[ -n "$CONFIG" ]] || { echo "--config is required; inspect the OpenPI checkout for the current Pi0 training config." >&2; exit 2; }
[[ -f "$OPENPI_ROOT/scripts/train.py" ]] || { echo "OpenPI training entry point not found under $OPENPI_ROOT" >&2; exit 1; }

cd "$OPENPI_ROOT"
uv run scripts/train.py "$CONFIG" --exp-name "$EXP_NAME" --overwrite "${EXTRA_ARGS[@]}"