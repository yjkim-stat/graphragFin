"""Run the finance GraphRAG indexing workflow and evaluate the Knowledge Graph.

This script mirrors the finance experiment runner but stops after the indexing
stage. It measures wall-clock runtime, inspects the generated Knowledge Graph
artifacts, and derives heuristics that approximate entity/relation extraction
quality so that indexing configurations can be compared objectively.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from pprint import pformat
from typing import Any, Iterable, Optional

import numpy as np

from graphrag.api.index import build_index
from graphrag.config.enums import IndexingMethod
from graphrag.language_model.providers.huggingface.chat_model import (
    HuggingFaceChatModel,
)
from script_utils.utils_eval import (
    _entity_identifier_sets,
    _entities_by_document,
    _evaluate_entities,
    _evaluate_relationships,
    _evaluate_ner,
    _load_finance_documents,
)
from script_utils.utils_run import (
    _build_config,
    _ensure_workspace,
    _graph_summary,
    _json_default,
    _load_parquet,
    _setup_logger,
)
 

LOGGER_NAME = "finance_graphrag_index_eval"
logger = logging.getLogger(LOGGER_NAME)


def _serialise_args(args: argparse.Namespace) -> dict[str, Any]:
    """Return a JSON-serialisable snapshot of relevant CLI arguments."""

    excluded_keys = {"huggingface_token"}
    serialised: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in excluded_keys:
            continue
        if isinstance(value, Path):
            serialised[key] = str(value)
        elif isinstance(value, (list, tuple)):
            serialised[key] = list(value)
        else:
            serialised[key] = value
    return serialised


def _summarise_workflow_results(
    pipeline_results: list,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for result in pipeline_results:
        summaries.append(
            {
                "workflow": getattr(result, "workflow", None),
                "errors": [str(err) for err in getattr(result, "errors", []) or []],
            }
        )
    return summaries


def _evaluate_indexing_run(
    stats_data: dict[str, Any],
    wall_time_seconds: float,
    llm_usage_total: dict[str, int] | None = None,
    llm_usage_by_model: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    workflows = stats_data.get("workflows", {}) if stats_data else {}
    metrics: dict[str, Any] = {
        "wall_time_seconds": float(wall_time_seconds),
        "reported_total_runtime_seconds": float(stats_data.get("total_runtime", 0.0)),
        "input_load_time_seconds": float(stats_data.get("input_load_time", 0.0)),
        "workflow_durations_seconds": workflows,
    }

    if llm_usage_total or llm_usage_by_model:
        metrics["llm_usage"] = {
            "total": llm_usage_total or {},
            "by_model": llm_usage_by_model or {},
        }

    return metrics


async def _run_indexing_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    root_dir = args.workspace_dir.resolve()
    _ensure_workspace(root_dir)
    run_logger = _setup_logger(
        logging.getLogger(LOGGER_NAME),
        root_dir / "logs",
        "run_finance_indexing_evaluation.log",
    )
    run_logger.info(
        "Starting finance GraphRAG indexing evaluation", extra={"dataset": args.dataset_name}
    )

    if args.skip_community_reports:
        run_logger.info(
            "Community report workflows disabled; skipping community summarization stage."
        )

    token = args.huggingface_token or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError(
            "A Hugging Face API token is required. Provide it via --huggingface-token or set HUGGINGFACEHUB_API_TOKEN."
        )

    config = _build_config(args, root_dir, token)
    logger.info("Config:\n%s", pformat(config))
    run_logger.info(f'args.dataset_name:{args.dataset_name}')

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
    run_logger.info(f"sorted(ground_truth_entities):{ground_truth_entities}")
    run_logger.info(f"documents.iloc[0].to_dict():{pformat(documents.iloc[0].to_dict())}")
    run_logger.info(f"documents.iloc[0]['text']:{documents.iloc[0]['text']}")

    indexing_method = IndexingMethod(args.indexing_method)
    run_logger.info("Indexing method: %s", indexing_method)

    HuggingFaceChatModel.reset_usage()

    start_time = time.perf_counter()
    pipeline_results = await build_index(
        config=config,
        method=indexing_method,
        input_documents=documents,
    )
    wall_time_seconds = time.perf_counter() - start_time
    run_logger.info("Indexing wall time: %.2fs", wall_time_seconds)

    workflow_errors = {
        result.workflow: [str(error) for error in result.errors or []]
        for result in pipeline_results
        if getattr(result, "errors", None)
    }
    if workflow_errors:
        formatted_errors = ", ".join(
            f"{workflow}: {errors}" for workflow, errors in workflow_errors.items()
        )
        raise RuntimeError(
            "One or more GraphRAG workflows failed. "
            "Review the logs for details and address the underlying errors before rerunning the pipeline. "
            f"Reported failures: {formatted_errors}"
        )

    output_dir = Path(config.output.base_dir)

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
            f"Missing expected GraphRAG outputs: {', '.join(sorted(missing_required))}"
        )

    stats_path = output_dir / "stats.json"
    stats_data = (
        json.loads(stats_path.read_text(encoding="utf-8"))
        if stats_path.exists()
        else {}
    )

    llm_usage_by_model = HuggingFaceChatModel.get_usage()
    llm_usage_total = HuggingFaceChatModel.get_total_usage()
    if llm_usage_total.get("llm_calls", 0):
        run_logger.info("Aggregated LLM usage: %s", llm_usage_total)

    entity_metrics = _evaluate_entities(artifacts["entities"], artifacts["relationships"])
    relationship_metrics = _evaluate_relationships(
        artifacts["relationships"],
        _entity_identifier_sets(artifacts["entities"]),
    )

    predicted_entities = _entities_by_document(
        artifacts["entities"],
        artifacts["text_units"],
    )
    ner_metrics = _evaluate_ner(ground_truth_entities, predicted_entities)
    entity_metrics["ground_truth_comparison"] = ner_metrics

    graph_info = _graph_summary(
        {
            key: df
            for key, df in artifacts.items()
            if df is not None and key != "documents"
        },
        top_k=args.graph_top_k,
    )

    indexing_metrics = _evaluate_indexing_run(
        stats_data,
        wall_time_seconds,
        llm_usage_total=llm_usage_total,
        llm_usage_by_model=llm_usage_by_model,
    )

    workflow_summaries = _summarise_workflow_results(pipeline_results)

    evaluation_report = {
        "dataset": dataset_summary,
        "indexing": indexing_metrics,
        "entity_extraction": entity_metrics,
        "relationship_extraction": relationship_metrics,
        "graph": graph_info,
        "workflows": workflow_summaries,
        "artifact_paths": {
            name: (
                str((output_dir / f"{name}.parquet").resolve())
                if artifacts.get(name) is not None and name != "community_reports"
                else (
                    str((output_dir / "community_reports.parquet").resolve())
                    if name == "community_reports" and artifacts.get(name) is not None
                    else None
                )
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
        "arguments": _serialise_args(args),
    }

    evaluation_report["entity_ground_truth"] = ner_metrics

    run_logger.info("Indexing evaluation completed successfully")
    run_logger.debug("Evaluation report:\n%s", pformat(evaluation_report))
    return evaluation_report


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
        help=(
            "Hugging Face dataset identifier or path to a local CSV/JSON/JSONL file to load."
        ),
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
            "When provided, the script computes NER precision/recall against GraphRAG outputs."
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
        "--output-file",
        type=Path,
        default=Path("finance_graphrag_indexing_report.json"),
        help="Path where the JSON evaluation report will be written.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    report = asyncio.run(_run_indexing_evaluation(args))
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    logger.info("Saved indexing evaluation report to %s", args.output_file)
    print(f"Saved indexing evaluation report to {args.output_file}")


def set_seed(seed):
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
