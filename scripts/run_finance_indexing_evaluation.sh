export HF_TOKEN=TODO
export CACHE_DIR=TODO
export ATTN_IMPLEMENTATION="sdpa"

#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper that forwards all arguments to the Python runner.
# Example:
#   ./scripts/run_finance_graphrag.sh --max-documents 100 --query "올해 주요 리스크는?"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)


N_SAMPLES=2

# DATASET_NAME=thunlp/docred
# CUDA_VISIBLE_DEVICES=0 python "${PROJECT_ROOT}/scripts/run_finance_indexing_evaluation.py" \
#     --workspace-dir /home/yjkim/gragfin-ws-sample${N_SAMPLES}\
#     --dataset-name ${DATASET_NAME} \
#     --skip-community-reports \
#     --split "train" \
#     --max-documents 200 \
#     --debug-document-limit $N_SAMPLES \
#     --model-name "meta-llama/Llama-3.2-1B-Instruct" \
#     --embedding-model-name "intfloat/multilingual-e5-base" \
#     --huggingface-task "text-generation" \
#     --encoding-model "cl100k_base" \
#     --embedding-encoding-model "cl100k_base" \
#     --max-new-tokens 1024 \
#     --temperature 0.2 \
#     --huggingface-token "${HF_TOKEN}" \
#     --graph-top-k 5 \
#     --indexing-method "standard" \
#     --output-file "/home/yjkim/gragfin/outputs/finance_kg_report-sample${N_SAMPLES}.json"\


DATASET_NAME=nlpaueb/finer-139
CUDA_VISIBLE_DEVICES=0 python "${PROJECT_ROOT}/scripts/run_finance_indexing_evaluation.py" \
    --workspace-dir /home/yjkim/gragfin-kg-ws-sample${N_SAMPLES}\
    --dataset-name ${DATASET_NAME} \
    --skip-community-reports \
    --split "train" \
    --max-documents 200 \
    --debug-document-limit $N_SAMPLES \
    --text-column tokens\
    --ground-truth-entity-column ner_tags\
    --model-name "meta-llama/Llama-3.2-1B-Instruct" \
    --embedding-model-name "intfloat/multilingual-e5-base" \
    --huggingface-task "text-generation" \
    --encoding-model "cl100k_base" \
    --embedding-encoding-model "cl100k_base" \
    --max-new-tokens 1024 \
    --temperature 0.2 \
    --huggingface-token "${HF_TOKEN}" \
    --graph-top-k 5 \
    --indexing-method "standard" \
    --output-file "/home/yjkim/gragfin/outputs/finance_kg_report-sample${N_SAMPLES}.json"\

