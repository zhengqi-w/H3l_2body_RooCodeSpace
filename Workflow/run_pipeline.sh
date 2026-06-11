#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODESPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GENERAL_CFG="$CODESPACE_DIR/configs/general_config.json"

is_bool_word() {
  local v
  v="$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')"
  [[ "$v" == "true" || "$v" == "false" ]]
}

RAW_CFG_PATH="${1:-$GENERAL_CFG}"
ARG2="${2:-all}"
ARG3="${3:-false}"

if is_bool_word "$ARG2" && [[ $# -lt 3 ]]; then
  STAGE="all"
  DRY_RUN="$ARG2"
else
  STAGE="$ARG2"
  DRY_RUN="$ARG3"
fi

if [[ -f "$RAW_CFG_PATH" ]]; then
  CFG_PATH="$(cd "$(dirname "$RAW_CFG_PATH")" && pwd -P)/$(basename "$RAW_CFG_PATH")"
elif [[ -f "$SCRIPT_DIR/$RAW_CFG_PATH" ]]; then
  CFG_PATH="$(cd "$(dirname "$SCRIPT_DIR/$RAW_CFG_PATH")" && pwd -P)/$(basename "$SCRIPT_DIR/$RAW_CFG_PATH")"
elif [[ -f "$CODESPACE_DIR/$RAW_CFG_PATH" ]]; then
  CFG_PATH="$(cd "$(dirname "$CODESPACE_DIR/$RAW_CFG_PATH")" && pwd -P)/$(basename "$CODESPACE_DIR/$RAW_CFG_PATH")"
else
  echo "[run_pipeline] Error: config file not found: $RAW_CFG_PATH" >&2
  exit 2
fi

if [[ "$CFG_PATH" != "$GENERAL_CFG" ]]; then
  echo "[run_pipeline] Info: default config is configs/general_config.json; using config: $CFG_PATH"
fi

if [[ "$STAGE" != "all" && "$STAGE" != "train" && "$STAGE" != "wp" && "$STAGE" != "analysis" ]]; then
  echo "[run_pipeline] Error: invalid stage '$STAGE' (use: train|wp|analysis|all)" >&2
  exit 3
fi

ROOT_CMD="${ROOT_EXECUTABLE:-}"
if [[ -z "$ROOT_CMD" ]]; then
  if command -v root >/dev/null 2>&1; then
    ROOT_CMD="$(command -v root)"
  elif [[ -x "/opt/anaconda3/envs/MLenv/bin/root" ]]; then
    ROOT_CMD="/opt/anaconda3/envs/MLenv/bin/root"
  else
    echo "[run_pipeline] Error: ROOT not found. Set ROOT_EXECUTABLE or add root to PATH." >&2
    exit 127
  fi
fi

ROOT_DIR="$(cd "$(dirname "$ROOT_CMD")" && pwd -P)"
export PATH="$ROOT_DIR:$PATH"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/h3l_matplotlib_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/h3l_xdg_cache}"
export ROOT_RDF_SNAPSHOT_INFO="${ROOT_RDF_SNAPSHOT_INFO:-0}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

cd "$CODESPACE_DIR"
echo "[run_pipeline] Config: $CFG_PATH"
echo "[run_pipeline] Stage : $STAGE"
echo "[run_pipeline] DryRun: $DRY_RUN"
"$ROOT_CMD" -l -b -q "Workflow/RunUnifiedPipeline.C(\"$CFG_PATH\", \"$STAGE\", $DRY_RUN)"
