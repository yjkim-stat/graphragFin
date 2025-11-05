"""Finance corpus HippoRAG experiment runner with knowledge graph evaluation."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from pprint import pformat
from typing import Any, Iterable, Optional

import numpy as np

from graphrag.api.index import build_index
from graphrag.config.embeddings import entity_description_embedding
from graphrag.config.enums import IndexingMethod
from graphrag.query.factory import get_local_search_engine
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.utils.api import get_embedding_store, reformat_context_data

# from scripts.run_finance_graphrag import (
from script_utils.utils_run import (
    _build_config,
    _compute_cost,
    _ensure_workspace,
    _graph_summary,
    _json_default,
    _load_parquet,
    _setup_logger,
)
from script_utils.utils_eval import (
    _load_finance_documents,
    _evaluate_entities,
    _evaluate_relationships,
    _entity_identifier_sets,
    _entities_by_document,
    _evaluate_ner
)

LOGGER_NAME = "finance_hipporag"
logger = logging.getLogger(LOGGER_NAME)


async def _run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    root_dir = args.workspace_dir.resolve()
    _ensure_workspace(root_dir)
    run_logger = _setup_logger(root_dir / "logs")
    run_logger.info(
        "Starting finance HippoRAG pipeline", extra={"dataset": args.dataset_name}
    )

    token = args.huggingface_token or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError(
            "A Hugging Face API token is required. Provide it via --huggingface-token or set HUGGINGFACEHUB_API_TOKEN."
        )

    args.indexing_method = IndexingMethod.Hippo.value
    config = _build_config(args, root_dir, token)
    logger.info("config:\n%s", config)

    (
        documents,
        dataset_summary,
        ground_truth_entities,
    ) = _load_finance_documents(
        dataset_name=args.dataset_name,
        split=args.split,
        max_documents=args.max_documents,
        debug_document_limit=args.debug_document_limit,
        text_column=args.text_column,
        title_column=args.title_column,
        metadata_columns=args.metadata_columns,
        entity_column=args.ground_truth_entity_column,
    )

    run_logger.info(
        "Loaded dataset '%s' split '%s' with %s documents (max=%s, debug_limit=%s)",
        args.dataset_name,
        args.split,
        len(documents),
        args.max_documents,
        args.debug_document_limit,
    )

    pipeline_results = await build_index(
        config=config,
        method=IndexingMethod.Hippo,
        input_documents=documents,
    )
    output_dir = Path(config.output.base_dir)

    workflow_errors = {
        result.workflow: [str(error) for error in result.errors or []]
        for result in pipeline_results
        if result.errors
    }
    if workflow_errors:
        formatted_errors = ", ".join(
            f"{workflow}: {errors}" for workflow, errors in workflow_errors.items()
        )
        raise RuntimeError(
            "One or more GraphRAG workflows failed. "
            "Review the logs for details and address the underlying errors before rerunning "
            f"the pipeline. Reported failures: {formatted_errors}"
        )

    community_reports_df = None
    if not args.skip_community_reports:
        community_reports_df = _load_parquet(output_dir, "community_reports.parquet")

    artifacts = {
        "documents": _load_parquet(output_dir, "documents.parquet"),
        "entities": _load_parquet(output_dir, "entities.parquet"),
        "relationships": _load_parquet(output_dir, "relationships.parquet"),
        "communities": _load_parquet(output_dir, "communities.parquet"),
        "community_reports": community_reports_df,
        "text_units": _load_parquet(output_dir, "text_units.parquet"),
        "covariates": _load_parquet(output_dir, "covariates.parquet"),
    }

    if artifacts["text_units"] is not None and "document_ids" in artifacts["text_units"].columns:
        artifacts["text_units"]["document_ids"] = artifacts["text_units"][
            "document_ids"
        ].astype(str)

    required_artifacts = [
        "documents",
        "entities",
        "relationships",
        "communities",
        "text_units",
    ]
    if not args.skip_community_reports:
        required_artifacts.append("community_reports")

    missing_required = [
        name for name in required_artifacts if artifacts.get(name) is None
    ]
    if missing_required:
        raise FileNotFoundError(
            f"Missing expected GraphRAG outputs: {', '.join(missing_required)}"
        )

    stats_path = output_dir / "stats.json"
    stats_data = (
        json.loads(stats_path.read_text(encoding="utf-8"))
        if stats_path.exists()
        else {}
    )
    context_state_path = output_dir / "context.json"
    context_state = (
        json.loads(context_state_path.read_text(encoding="utf-8"))
        if context_state_path.exists()
        else {}
    )

    vector_store_args = {
        name: store.model_dump() for name, store in config.vector_store.items()
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

    if args.skip_community_reports or artifacts["community_reports"] is None:
        reports = []
    else:
        reports = read_indexer_reports(
            artifacts["community_reports"],
            artifacts["communities"],
            args.community_level,
            config=config,
        )

    search_engine = get_local_search_engine(
        config=config,
        reports=reports,
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
        response_text = json.dumps(response_text, ensure_ascii=False)

    entity_metrics = _evaluate_entities(
        artifacts["entities"], artifacts["relationships"]
    )
    relationship_metrics = _evaluate_relationships(
        artifacts["relationships"],
        _entity_identifier_sets(artifacts["entities"])
        if artifacts["entities"] is not None
        else {},
    )
    predicted_entities = _entities_by_document(
        artifacts["entities"], artifacts["text_units"]
    )
    ner_metrics = _evaluate_ner(ground_truth_entities, predicted_entities)
    entity_metrics["ground_truth_comparison"] = ner_metrics

    evaluation_summary = {
        "entity_extraction": entity_metrics,
        "relationship_extraction": relationship_metrics,
        "entity_ground_truth": ner_metrics,
    }

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
        "evaluation": evaluation_summary,
    }

    run_logger.info("HippoRAG pipeline completed successfully")
    run_logger.debug("HippoRAG report:\n%s", pformat(report))
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
        "--skip-community-reports",
        action="store_true",
        help="Disable community report workflows and omit community report artifacts.",
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
        "--debug-document-limit",
        type=int,
        default=None,
        help="Optional limit to reduce the dataset size when debugging.",
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
        "--ground-truth-entity-column",
        default=None,
        help=(
            "Dataset column containing ground truth entity annotations for each document. "
            "If omitted, built-in heuristics attempt to derive them."
        ),
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
        help=(
            "Hugging Face inference task for the chat model (e.g. text-generation, chat-completion)."
        ),
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
        default=Path("finance_hipporag_report.json"),
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
    logger.info("Saved report to %s", args.output_file)
    print(f"Saved report to {args.output_file}")


def set_seed(seed: int) -> None:
    import random
    import torch
    import torch.backends.cudnn as cudnn

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)


if __name__ == "__main__":
    set_seed(123)
    main()
