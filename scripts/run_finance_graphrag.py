"""Finance corpus end-to-end GraphRAG experiment runner.

This script loads the ``AnonymousLLMer/finance-corpus-krx`` dataset from
Hugging Face, builds a GraphRAG index using the Hugging Face inference
endpoints, runs a local-search style generation, and stores the resulting
artifacts (retrieved context, graph summary, and usage metrics) in a JSON
report. The script exposes a CLI so experiments can be repeated with different
models or configuration parameters.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from datasets import load_dataset

from graphrag.api.index import build_index
from graphrag.config.create_graphrag_config import create_graphrag_config
from graphrag.config.embeddings import entity_description_embedding
from graphrag.config.enums import IndexingMethod
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.query.factory import get_local_search_engine
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.utils.api import get_embedding_store, reformat_context_data


def _json_default(value: Any) -> Any:
    """Convert unsupported JSON types to serialisable forms."""

    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.to_dict(orient="records")
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


def _ensure_workspace(root_dir: Path) -> None:
    """Create the directory tree expected by GraphRAG."""

    root_dir.mkdir(parents=True, exist_ok=True)
    for relative in [
        "cache",
        "logs",
        "output",
        "output/lancedb",
        "update_output",
    ]:
        (root_dir / relative).mkdir(parents=True, exist_ok=True)


def _guess_text_column(example: dict[str, Any]) -> str:
    for key, value in example.items():
        if isinstance(value, str) and value.strip():
            return key
    msg = "Unable to infer a text column from the dataset example."
    raise ValueError(msg)


def _guess_title_column(example: dict[str, Any], text_column: str) -> Optional[str]:
    for key, value in example.items():
        if key == text_column:
            continue
        if isinstance(value, str) and value.strip():
            return key
    return None


def _load_finance_documents(
    dataset_name: str,
    split: str,
    max_documents: Optional[int],
    text_column: Optional[str],
    title_column: Optional[str],
    metadata_columns: Optional[list[str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the finance dataset and normalise it into a DataFrame."""

    dataset = load_dataset(dataset_name, split=split)
    if len(dataset) == 0:
        raise ValueError("Loaded dataset is empty.")

    first_example = dataset[0]
    resolved_text_column = text_column or _guess_text_column(first_example)
    resolved_title_column = title_column or _guess_title_column(
        first_example, resolved_text_column
    )

    rows: list[dict[str, Any]] = []
    text_lengths: list[int] = []

    limit = max_documents if max_documents is not None else len(dataset)
    limit = min(limit, len(dataset))
    for idx in range(limit):
        example = dataset[idx]
        text = str(example[resolved_text_column])
        title = (
            str(example[resolved_title_column])
            if resolved_title_column is not None
            else f"Document {idx}"
        )
        text_lengths.append(len(text))

        metadata: dict[str, Any] | None = None
        selected_metadata = metadata_columns or []
        if selected_metadata:
            metadata = {
                key: example.get(key) for key in selected_metadata if key in example
            }

        rows.append(
            {
                "id": example.get("id", idx),
                "title": title,
                "text": text,
                "metadata": metadata,
            }
        )

    documents = pd.DataFrame(rows)
    dataset_summary = {
        "dataset": dataset_name,
        "split": split,
        "document_count": len(documents),
        "text_column": resolved_text_column,
        "title_column": resolved_title_column,
        "text_length": {
            "min": int(min(text_lengths)),
            "max": int(max(text_lengths)),
            "mean": float(sum(text_lengths) / len(text_lengths)),
        },
    }
    if metadata_columns:
        dataset_summary["metadata_columns"] = metadata_columns

    return documents, dataset_summary


def _build_config(args: argparse.Namespace, root_dir: Path, token: str) -> GraphRagConfig:
    """Create a GraphRAG configuration for Hugging Face providers."""

    config_dict: dict[str, Any] = {
        "root_dir": str(root_dir),
        "models": {
            "default_chat_model": {
                "type": "huggingface_chat",
                "model": args.model_name,
                "api_key": token,
                "encoding_model": args.encoding_model,
                "huggingface_task": args.huggingface_task,
                "huggingface_parameters": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                },
            },
            "default_embedding_model": {
                "type": "huggingface_embedding",
                "model": args.embedding_model_name,
                "api_key": token,
                "encoding_model": args.embedding_encoding_model,
                "huggingface_parameters": {
                    "normalize": True,
                },
            },
        },
        "cache": {"type": "file", "base_dir": "cache"},
        "reporting": {"type": "file", "base_dir": "logs"},
        "output": {"type": "file", "base_dir": "output"},
        "update_index_output": {"type": "file", "base_dir": "update_output"},
        "input": {
            "storage": {"type": "memory", "base_dir": "input"},
            "file_type": "text",
            "text_column": "text",
            "title_column": "title",
            "metadata": ["metadata"],
        },
        "vector_store": {
            "default_vector_store": {
                "type": "lancedb",
                "db_uri": str((root_dir / "output" / "lancedb").resolve()),
                "container_name": "default",
                "overwrite": True,
            }
        },
    }
    return create_graphrag_config(config_dict, root_dir=str(root_dir))


def _load_parquet(output_dir: Path, filename: str) -> Optional[pd.DataFrame]:
    file_path = output_dir / filename
    if not file_path.exists():
        return None
    return pd.read_parquet(file_path)


def _graph_summary(artifacts: dict[str, pd.DataFrame], top_k: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "counts": {name: len(df) for name, df in artifacts.items()},
    }
    entities = artifacts.get("entities")
    if entities is not None and "degree" in entities.columns:
        top_entities = (
            entities.nlargest(min(top_k, len(entities)), "degree")
            .loc[:, [col for col in ["id", "title", "degree"] if col in entities.columns]]
            .to_dict(orient="records")
        )
        summary["top_entities_by_degree"] = top_entities
    relationships = artifacts.get("relationships")
    if relationships is not None:
        cols = [c for c in ["source", "target", "description", "weight"] if c in relationships.columns]
        summary["sample_relationships"] = (
            relationships.head(min(top_k, len(relationships))).loc[:, cols].to_dict(orient="records")
            if cols
            else []
        )
    communities = artifacts.get("communities")
    if communities is not None:
        for col in ["level", "size"]:
            if col in communities.columns:
                summary.setdefault("communities", {})[f"{col}_distribution"] = (
                    communities[col].value_counts().to_dict()
                )
    return summary


def _compute_cost(
    prompt_tokens: int,
    completion_tokens: int,
    prompt_cost_per_1k: float,
    completion_cost_per_1k: float,
) -> float:
    return (
        (prompt_tokens / 1000.0) * prompt_cost_per_1k
        + (completion_tokens / 1000.0) * completion_cost_per_1k
    )


async def _run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    root_dir = args.workspace_dir.resolve()
    _ensure_workspace(root_dir)

    token = args.huggingface_token or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError(
            "A Hugging Face API token is required. Provide it via --huggingface-token or set HUGGINGFACEHUB_API_TOKEN."
        )

    config = _build_config(args, root_dir, token)

    documents, dataset_summary = _load_finance_documents(
        dataset_name=args.dataset_name,
        split=args.split,
        max_documents=args.max_documents,
        text_column=args.text_column,
        title_column=args.title_column,
        metadata_columns=args.metadata_columns,
    )

    indexing_method = IndexingMethod(args.indexing_method)
    pipeline_results = await build_index(
        config=config,
        method=indexing_method,
        input_documents=documents,
    )

    output_dir = Path(config.output.base_dir)
    artifacts = {
        "documents": _load_parquet(output_dir, "documents.parquet"),
        "entities": _load_parquet(output_dir, "entities.parquet"),
        "relationships": _load_parquet(output_dir, "relationships.parquet"),
        "communities": _load_parquet(output_dir, "communities.parquet"),
        "community_reports": _load_parquet(output_dir, "community_reports.parquet"),
        "text_units": _load_parquet(output_dir, "text_units.parquet"),
        "covariates": _load_parquet(output_dir, "covariates.parquet"),
    }

    required_artifacts = [
        "documents",
        "entities",
        "relationships",
        "communities",
        "community_reports",
        "text_units",
    ]
    missing_required = [
        name for name in required_artifacts if artifacts.get(name) is None
    ]
    if missing_required:
        raise FileNotFoundError(
            f"Missing expected GraphRAG outputs: {', '.join(missing_required)}"
        )

    stats_path = output_dir / "stats.json"
    stats_data = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    context_state_path = output_dir / "context.json"
    context_state = (
        json.loads(context_state_path.read_text(encoding="utf-8"))
        if context_state_path.exists()
        else {}
    )

    vector_store_args = {
        name: store.model_dump()
        for name, store in config.vector_store.items()
    }

    description_embedding_store = get_embedding_store(
        config_args=vector_store_args,
        embedding_name=entity_description_embedding,
    )

    entities = read_indexer_entities(
        artifacts["entities"], artifacts["communities"], args.community_level
    )
    covariates = (
        {"claims": read_indexer_covariates(artifacts["covariates"])}
        if artifacts["covariates"] is not None
        else {}
    )
    search_engine = get_local_search_engine(
        config=config,
        reports=read_indexer_reports(
            artifacts["community_reports"],
            artifacts["communities"],
            args.community_level,
            config=config,
        ),
        text_units=read_indexer_text_units(artifacts["text_units"]),
        entities=entities,
        relationships=read_indexer_relationships(artifacts["relationships"]),
        covariates=covariates,
        description_embedding_store=description_embedding_store,
        response_type=args.response_type,
        system_prompt=None,
    )

    search_result = await search_engine.search(query=args.query)

    context_records = reformat_context_data(search_result.context_data)
    context_text = search_result.context_text
    if isinstance(context_text, list):
        context_text = "\n".join(context_text)

    prompt_tokens = search_result.prompt_tokens
    completion_tokens = search_result.output_tokens
    estimated_cost = _compute_cost(
        prompt_tokens,
        completion_tokens,
        args.prompt_cost_per_1k_tokens,
        args.completion_cost_per_1k_tokens,
    )

    graph_info = _graph_summary(
        {
            key: df
            for key, df in artifacts.items()
            if df is not None and key != "documents"
        },
        top_k=args.graph_top_k,
    )

    workflow_summaries = [
        {
            "workflow": result.workflow,
            "errors": [str(err) for err in result.errors] if result.errors else [],
        }
        for result in pipeline_results
    ]

    response_text = search_result.response
    if not isinstance(response_text, str):
        response_text = json.dumps(response_text, ensure_ascii=False, default=_json_default)

    report = {
        "query": args.query,
        "dataset": dataset_summary,
        "response": response_text,
        "response_metrics": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "llm_calls": search_result.llm_calls,
            "completion_time_seconds": search_result.completion_time,
            "context_character_count": len(context_text),
            "response_character_count": len(response_text),
            "estimated_cost": estimated_cost,
        },
        "retrieved_context": context_records,
        "graph": graph_info,
        "pipeline_stats": stats_data,
        "pipeline_state": context_state,
        "workflows": workflow_summaries,
        "artifact_paths": {
            name: (
                str((output_dir / f"{name}.parquet").resolve())
                if artifacts.get(name) is not None
                else None
            )
            for name in [
                "documents",
                "entities",
                "relationships",
                "communities",
                "community_reports",
                "text_units",
                "covariates",
            ]
        },
    }

    return report


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path("finance_workspace"),
        help="Directory where GraphRAG outputs and caches will be stored.",
    )
    parser.add_argument(
        "--dataset-name",
        default="AnonymousLLMer/finance-corpus-krx",
        help="Hugging Face dataset identifier to load.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=200,
        help="Maximum number of documents to index from the dataset.",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="Explicit text column name (auto-detected when omitted).",
    )
    parser.add_argument(
        "--title-column",
        default=None,
        help="Explicit title column name (auto-detected when omitted).",
    )
    parser.add_argument(
        "--metadata-columns",
        nargs="*",
        default=None,
        help="Optional metadata columns to preserve in the document store.",
    )
    parser.add_argument(
        "--model-name",
        default="HuggingFaceH4/zephyr-7b-beta",
        help="Hugging Face model for generation.",
    )
    parser.add_argument(
        "--embedding-model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face model used for embeddings.",
    )
    parser.add_argument(
        "--huggingface-task",
        default="text-generation",
        help="Hugging Face inference task for the chat model (e.g. text-generation, chat-completion).",
    )
    parser.add_argument(
        "--encoding-model",
        default="cl100k_base",
        help="Tokenizer encoding name to estimate prompt length for the chat model.",
    )
    parser.add_argument(
        "--embedding-encoding-model",
        default="cl100k_base",
        help="Tokenizer encoding name to estimate prompt length for the embedding model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens generated by the chat model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Generation temperature for the chat model.",
    )
    parser.add_argument(
        "--huggingface-token",
        default=None,
        help="Hugging Face API token (falls back to HUGGINGFACEHUB_API_TOKEN).",
    )
    parser.add_argument(
        "--query",
        default="요약된 최근 공시에서 주요 이슈는 무엇인가?",
        help="Natural language query to run against the graph.",
    )
    parser.add_argument(
        "--response-type",
        default="multiple paragraphs",
        help="Desired response style for the local search engine.",
    )
    parser.add_argument(
        "--community-level",
        type=int,
        default=2,
        help="Community level to target during retrieval.",
    )
    parser.add_argument(
        "--graph-top-k",
        type=int,
        default=5,
        help="Number of top entities and relationships to surface in the graph summary.",
    )
    parser.add_argument(
        "--indexing-method",
        choices=[method.value for method in IndexingMethod],
        default=IndexingMethod.Standard.value,
        help="GraphRAG indexing workflow to execute.",
    )
    parser.add_argument(
        "--prompt-cost-per-1k-tokens",
        type=float,
        default=0.0,
        help="Optional prompt cost (in currency units) per 1k tokens for cost estimation.",
    )
    parser.add_argument(
        "--completion-cost-per-1k-tokens",
        type=float,
        default=0.0,
        help="Optional completion cost (in currency units) per 1k tokens for cost estimation.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("finance_graphrag_report.json"),
        help="Path where the JSON report will be written.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    report = asyncio.run(_run_pipeline(args))
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"Saved report to {args.output_file}")


if __name__ == "__main__":
    main()
