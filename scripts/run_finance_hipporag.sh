export HF_TOKEN=TODO
export CACHE_DIR=TODO

#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper that forwards all arguments to the HippoRAG Python runner.
# Example:
#   ./scripts/run_finance_hipporag.sh --max-documents 100 --query "올해 주요 리스크는?"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

python "${PROJECT_ROOT}/scripts/run_finance_hipporag.py" \
    --workspace-dir /data/yjkim/gragfin \
    --dataset-name "AnonymousLLMer/finance-corpus-krx" \
    --split "train" \
    --max-documents 200 \
    --text-column "text" \
    --title-column "title" \
    --model-name "meta-llama/Llama-3.2-1B-Instruct" \
    --embedding-model-name "intfloat/multilingual-e5-base" \
    --huggingface-task "text-generation" \
    --encoding-model "cl100k_base" \
    --embedding-encoding-model "cl100k_base" \
    --max-new-tokens 1024 \
    --temperature 0.2 \
    --huggingface-token "${HF_TOKEN}" \
    --query "요약된 최근 공시에서 주요 이슈는 무엇인가?" \
    --response-type "multiple paragraphs" \
    --community-level 2 \
    --graph-top-k 5 \
    --prompt-cost-per-1k-tokens 0.0 \
    --completion-cost-per-1k-tokens 0.0 \
    --output-file "/home/yjkim/gragfin/outputs/finance_hipporag_report.json"
