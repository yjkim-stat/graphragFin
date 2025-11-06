export HF_TOKEN=TODO
export CACHE_DIR=TODO
export ATTN_IMPLEMENTATION="sdpa"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

GRAPHRAG_EXTRACT_GRAPH_PROMPT_CFG=extract_graph_financeV1
GRAPHRAG_EXTRACT_GRAPH_PROMPT_CFG_ABBR=financeV1
# GRAPHRAG_EXTRACT_GRAPH_PROMPT_CFG=extract_graph_default
# GRAPHRAG_EXTRACT_GRAPH_PROMPT_CFG_ABBR=default


LLM_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
LLM_MODEL_ABBR='llama3-8b-inst'
# LLM_MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
# LLM_MODEL_ABBR='llama3-3b-inst'
# LLM_MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# LLM_MODEL_ABBR='llama3-1b-inst'

EMB_MODEL_NAME="intfloat/multilingual-e5-base"
EMB_MODEL_ABBR='me5'

DATASET_NAME=TODO
DATASET_ABBR=TODO

DIR_NAME=Querying-${LLM_MODEL_ABBR}-${EMB_MODEL_ABBR}-Prompt${GRAPHRAG_EXTRACT_GRAPH_PROMPT_CFG_ABBR}

QUERY='What factors cause increases and decreases in copper futures prices, and what key factors influence copper futures price movements overall?'

indexing_method="standard" 
# indexing_method="fast" 

N_SAMPLES=8

CUDA_VISIBLE_DEVICES=0 python "${PROJECT_ROOT}/scripts/run_finance_graphrag.py" \
    --workspace-dir TODO\
    --dataset-name ${DATASET_NAME} \
    --skip-community-reports \
    --text-column "all_text" \
    --title-column 'title'\
    --max-documents 200 \
    --debug-document-limit $N_SAMPLES\
    --model-name $LLM_MODEL_NAME\
    --embedding-model-name $EMB_MODEL_NAME \
    --huggingface-task "text-generation" \
    --encoding-model "cl100k_base" \
    --embedding-encoding-model "cl100k_base" \
    --max-new-tokens 1024 \
    --temperature 0.2 \
    --huggingface-token "${HF_TOKEN}" \
    --query $QUERY \
    --response-type "multiple paragraphs" \
    --community-level 2 \
    --graph-top-k 5 \
    --indexing-method $indexing_method \
    --prompt-cost-per-1k-tokens 0.0 \
    --completion-cost-per-1k-tokens 0.0 \
    --output-file TODO
