"""Utility helpers shared across finance experiment scripts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from graphrag.config.create_graphrag_config import create_graphrag_config
from graphrag.config.embeddings import default_embeddings
from graphrag.config.enums import IndexingMethod
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.workflows.factory import PipelineFactory


def _json_default(value: Any) -> Any:
    """Convert unsupported JSON types to serialisable forms."""

    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.to_dict(orient="records")
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


def _setup_logger(
    logger: logging.Logger,
    log_dir: Path,
    log_filename: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure a logger with both file and stream handlers."""

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / log_filename

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    has_file_handler = any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", None) == str(log_file)
        for handler in logger.handlers
    )
    if not has_file_handler:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )
    if not has_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


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


def _resolve_workflow_overrides(
    indexing_method: str,
    skip_community_reports: bool,
) -> list[str] | None:
    """Return a workflow list with community report stages removed when requested."""

    if not skip_community_reports:
        return None

    try:
        method_enum = IndexingMethod(indexing_method)
    except ValueError:
        return None

    base_workflows = (
        PipelineFactory.pipelines.get(method_enum)
        or PipelineFactory.pipelines.get(method_enum.value)
        or []
    )
    if not base_workflows:
        return None

    skip_steps = {
        "create_community_reports",
        "create_community_reports_text",
        "update_community_reports",
    }
    filtered = [step for step in base_workflows if step not in skip_steps]

    if len(filtered) == len(base_workflows):
        return None

    return filtered


def _build_config(
    args: argparse.Namespace,
    root_dir: Path,
    token: str,
) -> GraphRagConfig:
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

    workflow_override = _resolve_workflow_overrides(
        args.indexing_method, args.skip_community_reports
    )
    if workflow_override is not None:
        config_dict["workflows"] = list(workflow_override)

    if args.skip_community_reports:
        community_free_embeddings = [
            name for name in default_embeddings if not name.startswith("community.")
        ]
        config_dict["embed_text"] = {
            "names": community_free_embeddings,
        }

    return create_graphrag_config(config_dict, root_dir=str(root_dir))


def _load_parquet(output_dir: Path, filename: str) -> Optional[pd.DataFrame]:
    """Safely load a parquet artifact, returning ``None`` when missing."""

    file_path = output_dir / filename
    if not file_path.exists():
        return None
    return pd.read_parquet(file_path)


def _graph_summary(
    artifacts: dict[str, pd.DataFrame],
    top_k: int,
) -> dict[str, Any]:
    """Generate a lightweight summary of key graph artifacts."""

    summary: dict[str, Any] = {
        "counts": {name: len(df) for name, df in artifacts.items()},
    }

    entities = artifacts.get("entities")
    if entities is not None and "degree" in entities.columns:
        top_entities = (
            entities.nlargest(min(top_k, len(entities)), "degree")
            .loc[
                :,
                [col for col in ["id", "title", "degree"] if col in entities.columns],
            ]
            .to_dict(orient="records")
        )
        summary["top_entities_by_degree"] = top_entities

    relationships = artifacts.get("relationships")
    if relationships is not None:
        cols = [
            col
            for col in ["source", "target", "description", "weight"]
            if col in relationships.columns
        ]
        summary["sample_relationships"] = (
            relationships.head(min(top_k, len(relationships))).loc[:, cols].to_dict(
                orient="records"
            )
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
    """Estimate an LLM invocation cost given token counts and pricing."""

    return (
        (prompt_tokens / 1000.0) * prompt_cost_per_1k
        + (completion_tokens / 1000.0) * completion_cost_per_1k
    )

