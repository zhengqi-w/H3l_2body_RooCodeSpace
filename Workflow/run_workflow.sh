#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODESPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CFG_PATH="${1:-$CODESPACE_DIR/configs/general_config.json}"

# ROOT executable resolution order:
# 1) ROOT_EXECUTABLE env var
# 2) root from PATH
# 3) conda fallback used by ProcessWorkflow
ROOT_CMD="${ROOT_EXECUTABLE:-}"
if [[ -z "$ROOT_CMD" ]]; then
	if command -v root >/dev/null 2>&1; then
		ROOT_CMD="$(command -v root)"
	elif [[ -x "/opt/anaconda3/envs/MLenv/bin/root" ]]; then
		ROOT_CMD="/opt/anaconda3/envs/MLenv/bin/root"
	else
		echo "[run_workflow] Error: ROOT not found. Set ROOT_EXECUTABLE or add 'root' to PATH." >&2
		exit 127
	fi
fi

cd "$CODESPACE_DIR"
echo "[run_workflow] Using config: $CFG_PATH"
echo "[run_workflow] Using ROOT executable: $ROOT_CMD"
"$ROOT_CMD" -l -b -q "Workflow/ProcessWorkflow.C(\"$CFG_PATH\")"
