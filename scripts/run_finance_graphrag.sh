#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper that forwards all arguments to the Python runner.
# Example:
#   ./scripts/run_finance_graphrag.sh --max-documents 100 --query "올해 주요 리스크는?"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

python "${PROJECT_ROOT}/scripts/run_finance_graphrag.py" "$@"
