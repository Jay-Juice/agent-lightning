#!/usr/bin/env bash
set -Eeuo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Short matched pilot; the full entrypoint is run_full_python_p2.sh.
export AGL_PI_TRAINING_STEPS="${AGL_PI_TRAINING_STEPS:-8}"
export AGL_SKIP_FINAL_EXPORT=1
exec bash "$TOOLS/run_full_python_p2.sh" "$@"
