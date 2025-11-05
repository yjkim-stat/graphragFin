# GRAGFin

GRAGFin is an extended version of Microsoft Research's GraphRAG tailored for financial-knowledge discovery and retrieval-augmented generation. Building on the original GraphRAG pipeline, GRAGFin introduces additional workflows, model integrations, and evaluation tools that make it easier to work with open-source large language models and to assess the quality of generated knowledge graphs.

## Key Features

- **Hugging Face language model provider** – Seamlessly run GraphRAG-style pipelines with open-source LLMs via the provider interfaces under `graphrag/language_model/providers/huggingface`.
- **Dual GraphRAG and HippoRAG workflows** – Execute either GraphRAG or HippoRAG style retrieval flows through the components in the `hipporag` package.
- **Rule-based knowledge graph annotation** – Evaluate and enrich knowledge graph indexing with the rule-based annotators in `indexing_utils/annotate-v2.py`, supporting lexicon priors, tokenization, and POS-tagging driven variants.

## Quickstart

The `scripts/` directory contains end-to-end shell scripts that demonstrate the recommended way to run GRAGFin:

- `scripts/run_finance_graphrag.sh` – Complete workflow from indexing through querying.
- `scripts/run_finance_indexing_evaluation.sh` – Run only the indexing pipeline and knowledge-graph evaluation.

Review each script for required environment variables and configuration paths before execution, then run them from the repository root, for example:

```bash
bash scripts/run_finance_graphrag.sh
```

## Documentation and Community Resources

- [GraphRAG documentation](https://microsoft.github.io/graphrag) – Core concepts that remain applicable to GRAGFin.
- [Microsoft Research blog post](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/) – Overview of the original GraphRAG project.

Please consult `DEVELOPING.md`, `CONTRIBUTING.md`, and `RAI_TRANSPARENCY.md` for development guidance, contribution instructions, and responsible AI considerations.

## Source Projects

GRAGFin incorporates code from the following upstream repositories:

- GraphRAG – <https://github.com/microsoft/graphrag>
- HippoRAG – <https://github.com/OSU-NLP-Group/HippoRAG>

## Citation

If you use GRAGFin in academic work or software, please cite both upstream projects in addition to this repository:

```
@software{graphrag,
  author  = {Dwivedi, Karan and team},
  title   = {GraphRAG},
  year    = {2024},
  url     = {https://github.com/microsoft/graphrag}
}

@software{hipporag,
  author  = {Yu, Dian and team},
  title   = {HippoRAG},
  year    = {2024},
  url     = {https://github.com/OSU-NLP-Group/HippoRAG}
}
```
