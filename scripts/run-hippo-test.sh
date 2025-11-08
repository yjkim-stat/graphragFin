export HF_TOKEN=TODO
export CACHE_DIR=TODO
export ATTN_IMPLEMENTATION="sdpa"

CUDA_VISIBLE_DEVICES=3 python run-hippo-test.py
