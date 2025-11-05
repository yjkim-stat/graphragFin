"""Synchronous variant of the finance GraphRAG indexing evaluation runner."""

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

import run_finance_indexing_evaluation as async_runner

logger = logging.getLogger(async_runner.LOGGER_NAME)


def _run_indexing_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Run the finance indexing evaluation pipeline without async/await."""

    root_dir = args.workspace_dir.resolve()
    async_runner._ensure_workspace(root_dir)
    run_logger = async_runner._setup_logger(
        logging.getLogger(async_runner.LOGGER_NAME),
        root_dir / "logs",
        "run_finance_indexing_evaluation.log",
    )
    run_logger.info(
        "Starting finance GraphRAG indexing evaluation",
        extra={"dataset": args.dataset_name},
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

    config = async_runner._build_config(args, root_dir, token)
    logger.info("Config:\n%s", pformat(config))
    run_logger.info(f"args.dataset_name:{args.dataset_name}")

    (
        documents,
        dataset_summary,
        ground_truth_entities,
    ) = async_runner._load_finance_documents(
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
    run_logger.info(
        f"sorted(ground_truth_entities):{sorted(ground_truth_entities)}"
    )
    run_logger.info(
        f"documents.iloc[0].to_dict():{pformat(documents.iloc[0].to_dict())}"
    )
    run_logger.info(
        f"documents.iloc[0]['text']:{pformat(documents.iloc[0]['text'])}"
    )

    indexing_method = async_runner.IndexingMethod(args.indexing_method)
    run_logger.info("Indexing method: %s", indexing_method)

    start_time = time.perf_counter()
    pipeline_results = asyncio.run(
        async_runner.build_index(
            config=config,
            method=indexing_method,
            input_documents=documents,
        )
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
        community_reports_df = async_runner._load_parquet(
            output_dir, "community_reports.parquet"
        )

    artifacts = {
        "documents": async_runner._load_parquet(output_dir, "documents.parquet"),
        "entities": async_runner._load_parquet(output_dir, "entities.parquet"),
        "relationships": async_runner._load_parquet(
            output_dir, "relationships.parquet"
        ),
        "communities": async_runner._load_parquet(output_dir, "communities.parquet"),
        "community_reports": community_reports_df,
        "text_units": async_runner._load_parquet(output_dir, "text_units.parquet"),
        "covariates": async_runner._load_parquet(output_dir, "covariates.parquet"),
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

    entity_metrics = async_runner._evaluate_entities(
        artifacts["entities"], artifacts["relationships"]
    )
    relationship_metrics = async_runner._evaluate_relationships(
        artifacts["relationships"],
        async_runner._entity_identifier_sets(artifacts["entities"]),
    )

    predicted_entities = async_runner._entities_by_document(
        artifacts["entities"],
        artifacts["text_units"],
    )
    ner_metrics = async_runner._evaluate_ner(
        ground_truth_entities, predicted_entities
    )
    entity_metrics["ground_truth_comparison"] = ner_metrics

    graph_info = async_runner._graph_summary(
        {
            key: df
            for key, df in artifacts.items()
            if df is not None and key != "documents"
        },
        top_k=args.graph_top_k,
    )

    indexing_metrics = async_runner._evaluate_indexing_run(
        stats_data, wall_time_seconds
    )

    workflow_summaries = async_runner._summarise_workflow_results(pipeline_results)

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
    }

    evaluation_report["entity_ground_truth"] = ner_metrics

    run_logger.info("Indexing evaluation completed successfully")
    run_logger.debug("Evaluation report:\n%s", pformat(evaluation_report))
    return evaluation_report


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = async_runner.parse_args(argv)
    report = _run_indexing_evaluation(args)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=async_runner._json_default),
        encoding="utf-8",
    )
    logger.info("Saved indexing evaluation report to %s", args.output_file)
    print(f"Saved indexing evaluation report to {args.output_file}")


if __name__ == "__main__":
    async_runner.set_seed(123)
    main()
