# This cell provides a reusable function to extract nested fields from a list of JSON-like results
# and convert them into a pandas DataFrame. A small demo dataset is included at the bottom.
#
# How to use with your own data:
# 1) Replace `results` with your actual list of dicts (or load from a JSON file).
# 2) Run the cell to get a DataFrame with the requested columns.
#
# If your data is a single dict that contains a list under some key (e.g., {"items": [...]}) 
# just pass that list to `extract_to_df(...)`.

from typing import Any, Iterable, Mapping, List, Dict
import pandas as pd
import json
from itertools import product
import os

# Helper to safely fetch nested keys like "a.b.c" from dicts
def get_nested(data: Mapping[str, Any], dotted_key: str, default=None) -> Any:
    cur = data
    for part in dotted_key.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            # print(f'cur:{cur}\npart:{part}')
            return 'Wrong'
    return cur

# Main utility: turn a list of results (dicts) into a DataFrame with selected columns
def extract_to_df(result: Iterable[Mapping[str, Any]], fields: List[str]) -> pd.DataFrame:
    row = {}
    for f in fields:
        row[f.split('.')[-1]] = get_nested(result, f, default=None)
    return row

# ---- Specify the fields you want to extract ----
FIELDS = [
    "indexing.wall_time_seconds",
    "indexing.llm_usage.by_model.extract_graph.prompt_tokens",
    "indexing.llm_usage.by_model.extract_graph.completion_tokens",
    "entity_extraction.ground_truth_comparison.micro.precision",
    "entity_extraction.ground_truth_comparison.micro.recall",
    "entity_extraction.ground_truth_comparison.micro.f1",
]

# prompts = ['default', 'PromptfinanceV1', 'PromptfinanceV2', 'PromptfinanceV3']
prompts = ['Promptdefault', 'PromptfinanceV1', 'PromptfinanceV2', 'PromptfinanceV3']
llm_models = ['llama3-1b-inst', 'llama3-3b-inst', 'llama3-8b-inst']
emb_models = ['me5']
n_samples = [2, 4, 8, 16]
dataset = 'news_annov2'
# dataset = 'docred'

results = []
for prompt, llm_model, emb_model, n_sample in product(prompts, llm_models, emb_models, n_samples):
    # prompt_name = prompt if prompt != 'default' else None
    folder_name = '-'.join([llm_model, emb_model, prompt])
    fpath = f'/home/yjkim/gragfin/outputs/{folder_name}/{dataset}-sample{n_sample}.json'
    # fpath = f'/home/yjkim/graphfin-results/{llm_model}-{emb_model}/{dataset}-sample{n_sample}.json'

    if not os.path.exists(fpath):  # ✅ 수정됨
        print(f'Skpped {fpath}')
        continue
                
    with open(fpath, "r", encoding="utf-8") as f:
        result = json.load(f)  # JSON이 list[dict] 형태여야 함
    result_dict = extract_to_df(result, FIELDS)
    result_dict['prompt'] = prompt
    result_dict['llm'] = llm_model
    result_dict['emb'] = emb_model
    result_dict['n'] = n_sample
    result_dict['dataset'] = dataset
    results.append(result_dict)

df = pd.DataFrame(results)
df = df.loc[:, ['dataset', 'prompt', 'llm', 'emb', 'n', 'prompt_tokens', 'completion_tokens', 'precision', 'recall', 'f1', 'wall_time_seconds']]
df['latency/sample'] = df['wall_time_seconds'] / df['n']
df['prompt/sample'] = df['prompt_tokens'] / df['n']
df['completion/sample'] = df['completion_tokens'] / df['n']
df.to_csv(f"/home/yjkim/gragfin/outputs/{dataset}-view.csv")
df.to_excel(f"/home/yjkim/gragfin/outputs/{dataset}-view.xlsx")
print(df)
