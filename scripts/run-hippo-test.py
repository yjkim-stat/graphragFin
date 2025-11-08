import os
from typing import List
import json
import argparse
import logging

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig
import pandas as pd


def main():
    # Prepare datasets and evaluation
    docs = [
        "Oliver Badman is a politician.",
        "George Rankin is a politician.",
        "Thomas Marwick is a politician.",
        "Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        "Erik Hort's birthplace is Montebello.",
        "Marina is bom in Minsk.",
        "Montebello is a part of Rockland County."
    ]

    save_dir = 'TODO'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = 'Transformers/meta-llama/Llama-3.1-8B-Instruct'  # Any OpenAI model name
    embedding_model_name = 'Transformers/intfloat/multilingual-e5-base'  # Embedding model name (NV-Embed, GritLM or Contriever for now)
    global_config = BaseConfig(
        save_dir=save_dir,
        openie_mode='Transformers-offline',
        information_extraction_model_name='Transformers/meta-llama/Llama-3.1-8B-Instruct'
    )

    # Startup a HippoRAG instance
    hipporag = HippoRAG(global_config,
                        save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        embedding_model_name=embedding_model_name,
                        )

    # Run indexing
    hipporag.index(docs=docs)
    
    # Separate Retrieval & QA
    queries = [
        "What is George Rankin's occupation?",
    ]

    retrieval_results, logs = hipporag.retrieve(queries=queries, num_to_retrieve=50, return_logs=True)
    retrieval_results = retrieval_results[0]
    with open(f'{hipporag.working_dir}/retrieval_results.txt', 'w') as f:
        for doc_i, doc in enumerate(retrieval_results.docs):
            f.write(f'\n Doc {doc_i+1} : {doc}\n')

    with open(f'{hipporag.working_dir}/logs.txt', 'w') as f:
        f.write('\n\n'.join(logs))


    graph_info_dict = hipporag.get_graph_info()
    with open(f'{hipporag.working_dir}/graph_info.json', 'w') as f:
        json.dump(graph_info_dict, fp=f, indent=4)
    hipporag.save_igraph()

    from igraph import Graph

    # 저장했던 그래프 불러오기
    g = Graph.Read_Pickle(f"{hipporag.working_dir}/graph.pickle")

    print(g)  # 노드/엣지 개수 확인

if __name__ == "__main__":
    main()