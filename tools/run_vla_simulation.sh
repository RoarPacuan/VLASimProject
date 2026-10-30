#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY=""
PI0_VARIANT="pi05"
EXPERIMENT_CONFIG=""
RUN_NAME=""
EPISODES=""
VIZ="none"
LIVESTREAM=""
PORT=""
SERVER_HOST="127.0.0.1"
MODEL_PATH="nvidia/GR00T-N1.6-DROID"
EMBODIMENT_TAG="OXE_DROID"
OUTPUT_BASE_DIR=""
RECORD_CAMERA_VIDEO=false
START_SERVER=true
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"
SERVER_PID=""
SERVER_LOG=""

usage() {
    cat <<EOF
Run a maintained Isaac Lab-Arena DROID experiment with Pi0 or GR00T.

Usage:
  $(basename "$0") --policy pi0|gr00t [options]

Options:
  --policy <name>             Policy to run (required).
    --pi0-variant <pi0|pi05>    OpenPI checkpoint variant (default: pi05).
  --experiment-config <path> Experiment YAML. Defaults to the maintained DROID config.
  --run-name <name>           Run name used for local overrides (required for custom YAML).
  --episodes <n>              Override the number of episodes.
  --viz <none|kit>            Visualization mode (default: none).
    --livestream <1|2>          Enable Isaac Sim WebRTC livestreaming.
  --port <port>               Local inference-server port (8000 for Pi0, 5555 for GR00T).
    --server-host <host>        External policy-server host (default: 127.0.0.1).
  --model-path <path>         GR00T model/checkpoint path.
  --embodiment-tag <tag>      GR00T embodiment tag (default: OXE_DROID).
  --output-base-dir <path>    Base directory for Arena evaluation outputs.
  --record-camera-video       Record camera videos in the Arena report.
    --external-server           Use an already-running policy server instead of Docker.
    SERVER_START_TIMEOUT=<sec>  Maximum server startup wait (default: 900).
  -h, --help                  Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy) POLICY="$2"; shift 2 ;;
        --pi0-variant) PI0_VARIANT="$2"; shift 2 ;;
        --experiment-config) EXPERIMENT_CONFIG="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        --episodes) EPISODES="$2"; shift 2 ;;
        --viz) VIZ="$2"; shift 2 ;;
        --livestream) LIVESTREAM="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --server-host) SERVER_HOST="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --embodiment-tag) EMBODIMENT_TAG="$2"; shift 2 ;;
        --output-base-dir) OUTPUT_BASE_DIR="$2"; shift 2 ;;
        --record-camera-video) RECORD_CAMERA_VIDEO=true; shift ;;
        --external-server) START_SERVER=false; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$POLICY" != "pi0" && "$POLICY" != "gr00t" ]]; then
    echo "--policy must be pi0 or gr00t" >&2
    usage >&2
    exit 2
fi

if [[ "$PI0_VARIANT" != "pi0" && "$PI0_VARIANT" != "pi05" ]]; then
    echo "--pi0-variant must be pi0 or pi05" >&2
    exit 2
fi

if [[ -n "$LIVESTREAM" && "$LIVESTREAM" != "1" && "$LIVESTREAM" != "2" ]]; then
    echo "--livestream must be 1 (public) or 2 (private/local network)" >&2
    exit 2
fi

if [[ -z "$EXPERIMENT_CONFIG" ]]; then
    if [[ "$POLICY" == "pi0" ]]; then
        EXPERIMENT_CONFIG="$ROOT_DIR/isaaclab_arena_environments/experiment_configs/droid_pnp_openpi_experiment.yaml"
        RUN_NAME="${RUN_NAME:-droid_pnp_openpi}"
        PORT="${PORT:-8000}"
    else
        EXPERIMENT_CONFIG="$ROOT_DIR/isaaclab_arena_environments/experiment_configs/droid_pnp_gr00t_experiment.yaml"
        RUN_NAME="${RUN_NAME:-droid_pnp_gr00t}"
        PORT="${PORT:-5555}"
    fi
elif [[ -z "$RUN_NAME" ]]; then
    echo "--run-name is required when --experiment-config is custom" >&2
    exit 2
fi

[[ -f "$EXPERIMENT_CONFIG" ]] || { echo "Experiment config not found: $EXPERIMENT_CONFIG" >&2; exit 1; }
if [[ "$START_SERVER" == true ]] && ! command -v docker >/dev/null 2>&1; then
    echo "Docker CLI is unavailable in this runtime. Start the policy server from the host and rerun with:" >&2
    echo "  --external-server --server-host 100.75.62.1" >&2
    exit 1
fi
runtime_has_isaaclab() {
    "$1" -c "import isaaclab" >/dev/null 2>&1
}

if [[ -z "${ARENA_PYTHON:-}" ]]; then
    if [[ -x "$ROOT_DIR/.venv/bin/python" ]] && runtime_has_isaaclab "$ROOT_DIR/.venv/bin/python"; then
        ARENA_PYTHON="$ROOT_DIR/.venv/bin/python"
    elif [[ -x /isaac-sim/python.sh ]] && runtime_has_isaaclab /isaac-sim/python.sh; then
        ARENA_PYTHON=/isaac-sim/python.sh
    else
        echo "No usable Isaac Lab runtime found. Run this launcher inside the Arena Docker container,"
        echo "or set ARENA_PYTHON to an executable that can import isaaclab." >&2
        exit 1
    fi
elif [[ ! -x "$ARENA_PYTHON" ]]; then
    echo "ARENA_PYTHON is not executable: $ARENA_PYTHON" >&2
    exit 1
elif ! runtime_has_isaaclab "$ARENA_PYTHON"; then
    echo "ARENA_PYTHON cannot import isaaclab: $ARENA_PYTHON" >&2
    exit 1
fi

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -INT "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [[ -n "$SERVER_LOG" ]]; then
        echo "Inference server log: $SERVER_LOG"
    fi
}
trap cleanup EXIT INT TERM

if [[ "$START_SERVER" == true ]]; then
    SERVER_LOG="$(mktemp "${TMPDIR:-/tmp}/arena-${POLICY}-server.XXXXXX.log")"
    if [[ "$POLICY" == "pi0" ]]; then
        (
            cd "$ROOT_DIR"
            ./isaaclab_arena_openpi/docker/run_openpi_server.sh -v "$PI0_VARIANT" -p "$PORT"
        ) >"$SERVER_LOG" 2>&1 &
    else
        GR00T_ROOT="$ROOT_DIR/submodules/Isaac-GR00T"
        [[ -d "$GR00T_ROOT" ]] || { echo "Initialize submodules/Isaac-GR00T first." >&2; exit 1; }
        (
            cd "$GR00T_ROOT"
            uv run python gr00t/eval/run_gr00t_server.py \
                --model-path "$MODEL_PATH" \
                --embodiment-tag "$EMBODIMENT_TAG" \
                --device cuda --host 127.0.0.1 --port "$PORT"
        ) >"$SERVER_LOG" 2>&1 &
    fi
    SERVER_PID=$!

    echo "Waiting for $POLICY server on 127.0.0.1:$PORT ..."
    for iteration in $(seq 1 "$SERVER_START_TIMEOUT"); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            cat "$SERVER_LOG" >&2
            exit 1
        fi
        if grep -qE "server listening on 0\.0\.0\.0:${PORT}|ready|Ready|listening" "$SERVER_LOG"; then
            break
        fi
        if (( iteration % 15 == 0 )); then
            echo "Server is still starting; latest log output:"
            tail -n 3 "$SERVER_LOG" || true
        fi
        sleep 1
    done
    if ! grep -qE "server listening on 0\.0\.0\.0:${PORT}|ready|Ready|listening" "$SERVER_LOG"; then
        echo "Timed out waiting for $POLICY server." >&2
        cat "$SERVER_LOG" >&2
        exit 1
    fi
else
    echo "Using external $POLICY server on $SERVER_HOST:$PORT ..."
    external_server_accepts_websocket() {
        local response
        if ! exec 3<>"/dev/tcp/$SERVER_HOST/$PORT"; then
            return 1
        fi
        printf 'GET / HTTP/1.1\r\nHost: %s\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: SGVsbG9XZWJTb2NrZXQ=\r\nSec-WebSocket-Version: 13\r\n\r\n' "$SERVER_HOST" >&3
        IFS= read -r -t 2 response <&3 || true
        exec 3>&-
        exec 3<&-
        [[ "$response" == HTTP/* ]]
    }

    for iteration in $(seq 1 120); do
        if external_server_accepts_websocket; then
            break
        fi
        if (( iteration % 15 == 0 )); then
            echo "Still waiting for external server on $SERVER_HOST:$PORT ..."
        fi
        sleep 1
    done
    if ! external_server_accepts_websocket; then
        echo "External server did not open $SERVER_HOST:$PORT within 120 seconds." >&2
        exit 1
    fi
fi

OVERRIDES=(
    "runs.${RUN_NAME}.policy.remote_host=${SERVER_HOST}"
    "runs.${RUN_NAME}.policy.remote_port=${PORT}"
)
if [[ "$POLICY" == "pi0" ]]; then
    OVERRIDES+=("runs.${RUN_NAME}.policy.policy_variant=${PI0_VARIANT}")
fi
if [[ -n "$EPISODES" ]]; then
    OVERRIDES+=("runs.${RUN_NAME}.rollout_limit.num_episodes=${EPISODES}")
fi
RUN_ARGS=(
    --experiment_config "$EXPERIMENT_CONFIG"
    --viz "$VIZ"
    "${OVERRIDES[@]}"
)
if [[ -n "$LIVESTREAM" ]]; then
    RUN_ARGS+=(
        --livestream "$LIVESTREAM"
        --kit_args "--enable omni.kit.livestream.webrtc"
    )
fi
if [[ -n "$OUTPUT_BASE_DIR" ]]; then
    RUN_ARGS+=(--output_base_dir "$OUTPUT_BASE_DIR")
fi
if [[ "$RECORD_CAMERA_VIDEO" == true ]]; then
    RUN_ARGS+=(--record_camera_video)
fi

echo "Running $POLICY with $EXPERIMENT_CONFIG"
cd "$ROOT_DIR"
PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y "$ARENA_PYTHON" \
    isaaclab_arena/evaluation/experiment_runner.py "${RUN_ARGS[@]}"