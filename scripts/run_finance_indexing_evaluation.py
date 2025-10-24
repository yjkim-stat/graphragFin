"""Run the finance GraphRAG indexing workflow and evaluate the Knowledge Graph.

This script mirrors the finance experiment runner but stops after the indexing
stage. It measures wall-clock runtime, inspects the generated Knowledge Graph
artifacts, and derives heuristics that approximate entity/relation extraction
quality so that indexing configurations can be compared objectively.
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from pprint import pformat
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from datasets import load_dataset

from graphrag.api.index import build_index
from graphrag.config.create_graphrag_config import create_graphrag_config
from graphrag.config.embeddings import default_embeddings
from graphrag.config.enums import IndexingMethod
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.workflows.factory import PipelineFactory
 

LOGGER_NAME = "finance_graphrag_index_eval"
logger = logging.getLogger(LOGGER_NAME)


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


def _setup_logger(log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure a module-level logger with a file handler."""

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run_finance_indexing_evaluation.log"

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


def _is_docred_like_example(example: dict[str, Any]) -> bool:
    """Heuristically detect DocRED-style token/vertex structured examples."""

    sentences = example.get("sents")
    vertex_set = example.get("vertexSet")
    if not isinstance(sentences, list) or not isinstance(vertex_set, list):
        return False

    has_tokenised_sentence = any(
        isinstance(sentence, (list, tuple)) for sentence in sentences
    )
    has_entity_clusters = any(
        isinstance(cluster, (list, tuple)) for cluster in vertex_set
    )
    return has_tokenised_sentence and has_entity_clusters


def _is_token_tag_example(example: dict[str, Any]) -> bool:
    """Identify token-level NER examples with parallel token/tag sequences."""

    tokens = example.get("tokens")
    tags = example.get("ner_tags")

    if not isinstance(tokens, (list, tuple)) or not isinstance(tags, (list, tuple)):
        return False

    if not tokens or not tags or len(tokens) != len(tags):
        return False

    return True


def _docred_sentences_to_text(sentences: Any) -> str:
    """Join tokenised sentences into a newline-separated document string."""

    if not isinstance(sentences, list):
        return ""

    flattened: list[str] = []
    for sentence in sentences:
        if isinstance(sentence, (list, tuple)):
            tokens = [str(token) for token in sentence if token is not None]
            joined = " ".join(tokens).strip()
        else:
            joined = str(sentence).strip()
        if joined:
            flattened.append(joined)
    return "\n".join(flattened)


def _extract_docred_entities(vertex_set: Any) -> list[str]:
    """Collapse DocRED-style entity clusters into canonical surface forms."""

    if not isinstance(vertex_set, list):
        return []

    entities: list[str] = []
    seen: set[str] = set()
    for cluster in vertex_set:
        if not isinstance(cluster, (list, tuple)):
            continue

        canonical_name: Optional[str] = None
        for mention in cluster:
            if isinstance(mention, dict):
                name = mention.get("name")
            else:
                name = mention
            if not name:
                continue
            candidate = str(name).strip()
            if candidate:
                canonical_name = candidate
                break

        if not canonical_name:
            continue

        key = canonical_name.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append(canonical_name)

    return entities


def _preprocess_docred_like_example(
    example: dict[str, Any],
    default_title: str,
) -> dict[str, Any]:
    """Normalise DocRED-style examples for the finance indexing workflow."""

    text = _docred_sentences_to_text(example.get("sents"))
    raw_title = example.get("title")
    title = (
        str(raw_title).strip()
        if isinstance(raw_title, str) and str(raw_title).strip()
        else default_title
    )

    entities = _extract_docred_entities(example.get("vertexSet"))

    metadata: dict[str, Any] = {}
    if "labels" in example:
        metadata["labels"] = example["labels"]

    sentences = example.get("sents")
    if isinstance(sentences, list):
        metadata["sentence_count"] = sum(
            1 for sentence in sentences if isinstance(sentence, (list, tuple, str))
        )

    vertex_set = example.get("vertexSet")
    if isinstance(vertex_set, list):
        metadata["entity_cluster_count"] = sum(
            1 for cluster in vertex_set if isinstance(cluster, (list, tuple))
        )

    return {
        "title": title,
        "text": text,
        "entities": entities,
        "metadata": metadata or None,
    }


def _token_tag_label_lookup(feature: Any) -> tuple[dict[int, str] | None, list[str]]:
    """Extract the label names for Sequence(ClassLabel) style ner_tags features."""

    if feature is None:
        return None, []

    names: Optional[Iterable[str]] = None

    try:
        # Sequence(ClassLabel) exposes the underlying ClassLabel via ``feature``
        if hasattr(feature, "feature"):
            subfeature = getattr(feature, "feature")
        else:
            subfeature = feature
        names = getattr(subfeature, "names", None)
    except Exception:  # pragma: no cover - defensive against feature quirks
        names = None

    if not names:
        return None, []

    ordered_names = list(names)
    return {index: name for index, name in enumerate(ordered_names)}, ordered_names


def _detokenize(tokens: Iterable[str]) -> str:
    """Reconstruct text from space-separated tokens with simple punctuation rules."""

    result = ""
    previous_token = ""

    no_space_before = {
        ".",
        ",",
        ":",
        ";",
        "!",
        "?",
        "%",
        "''",
        "\"",
        "'",
        "”",
        "’",
        "»",
        ")",
        "]",
        "}",
    }
    contractions = {
        "'s",
        "'re",
        "'m",
        "'ve",
        "'d",
        "'ll",
        "n't",
    }
    no_space_after = {"(", "[", "{", "$", "``", "“", "‘", "«"}

    for token in tokens:
        if token is None:
            continue
        piece = str(token)
        if not piece:
            continue

        if not result:
            result = piece
        elif piece in no_space_before or piece in contractions or piece.startswith("'"):
            result += piece
        elif previous_token in no_space_after:
            result += piece
        else:
            result += f" {piece}"

        previous_token = piece

    return result


def _extract_token_tag_entities(
    tokens: Any,
    tags: Any,
    label_lookup: Optional[dict[int, str]] = None,
) -> tuple[list[str], int, dict[str, int]]:
    """Collapse BIO-style token/tag annotations into canonical entity strings."""

    token_list = list(tokens) if isinstance(tokens, (list, tuple)) else None
    tag_list = list(tags) if isinstance(tags, (list, tuple)) else None

    if token_list is None or tag_list is None:
        return [], 0, {}

    tokens = token_list
    tags = tag_list

    span_count = 0
    label_counts: dict[str, int] = defaultdict(int)
    entities: list[str] = []
    seen: set[str] = set()

    entity_tokens: list[str] = []
    entity_label: Optional[str] = None

    prefix_map = {"U": "S", "L": "E", "M": "I"}

    def flush() -> None:
        nonlocal entity_tokens, entity_label, span_count
        if entity_tokens:
            text = _detokenize(entity_tokens).strip()
            if text:
                span_count += 1
                if entity_label:
                    label_counts[entity_label] += 1
                key = text.lower()
                if key not in seen:
                    seen.add(key)
                    entities.append(text)
        entity_tokens = []
        entity_label = None

    for token, raw_tag in zip(tokens, tags):
        token_text = str(token).strip() if token is not None else ""
        if not token_text:
            flush()
            continue

        label_name: Optional[str] = None
        if label_lookup is not None:
            if isinstance(raw_tag, (int, np.integer)):
                label_name = label_lookup.get(int(raw_tag))
            elif isinstance(raw_tag, str) and raw_tag.isdigit():
                label_name = label_lookup.get(int(raw_tag))

        if label_name is None:
            label_name = str(raw_tag) if raw_tag is not None else "O"

        label_name = label_name.strip()
        if not label_name:
            label_name = "O"

        if label_name.upper() == "O":
            flush()
            continue

        if "-" in label_name:
            prefix, label = label_name.split("-", 1)
        else:
            prefix, label = label_name, label_name

        prefix = prefix_map.get(prefix.upper(), prefix.upper())
        label = label.strip() or label_name

        if prefix == "O":
            flush()
            continue

        if prefix in {"B", "S"} or entity_label != label:
            flush()
            entity_label = label
            entity_tokens = [token_text]
            if prefix in {"S", "E"}:
                flush()
        else:
            entity_tokens.append(token_text)
            if prefix == "E":
                flush()

    flush()

    return entities, span_count, dict(label_counts)


def _preprocess_token_tag_example(
    example: dict[str, Any],
    default_title: str,
    label_lookup: Optional[dict[int, str]] = None,
) -> dict[str, Any]:
    """Normalise token/tag style examples for the finance indexing workflow."""

    raw_tokens = example.get("tokens")
    tokens = list(raw_tokens) if isinstance(raw_tokens, (list, tuple)) else []
    token_strings = [str(token).strip() if token is not None else "" for token in tokens]
    text = _detokenize(token_strings).strip()

    raw_title = example.get("title")
    if isinstance(raw_title, str) and raw_title.strip():
        title = raw_title.strip()
    else:
        raw_id = example.get("id")
        if isinstance(raw_id, str) and raw_id.strip():
            title = raw_id.strip()
        elif isinstance(raw_id, (int, np.integer)):
            title = f"Document {raw_id}"
        else:
            title = default_title

    raw_tags = example.get("ner_tags")
    tag_list = list(raw_tags) if isinstance(raw_tags, (list, tuple)) else []

    entities, span_count, label_counts = _extract_token_tag_entities(
        token_strings, tag_list, label_lookup
    )

    metadata: dict[str, Any] = {
        "token_count": len(tokens),
        "tag_count": len(tag_list),
    }
    if entities:
        metadata["unique_entity_count"] = len(entities)
    if span_count:
        metadata["entity_span_count"] = span_count
    if label_counts:
        metadata["entity_label_counts"] = dict(sorted(label_counts.items()))

    return {
        "title": title,
        "text": text,
        "entities": entities,
        "metadata": metadata or None,
    }


def _normalise_creation_date(value: Any) -> str:
    """Convert a raw creation date value into an ISO-8601 string."""

    if isinstance(value, str):
        value = value.strip() or None

    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if timestamp is not None and not pd.isna(timestamp):
        return timestamp.isoformat()

    return pd.Timestamp.now(tz="UTC").isoformat()


def _load_finance_documents(
    dataset_name: str,
    split: str,
    max_documents: Optional[int],
    debug_document_limit: Optional[int],
    text_column: Optional[str],
    title_column: Optional[str],
    metadata_columns: Optional[list[str]],
    entity_column: Optional[str],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, set[str]]]:
    """Load the finance dataset and normalise it into a DataFrame."""
    dataset = load_dataset(dataset_name, split=split)
    if len(dataset) == 0:
        raise ValueError("Loaded dataset is empty.")

    first_example = dataset[0]
    docred_like = _is_docred_like_example(first_example)
    token_tag_like = False
    token_tag_label_lookup: dict[int, str] | None = None
    token_tag_label_names: list[str] = []

    if docred_like:
        resolved_text_column = text_column or "sents"
        resolved_title_column = title_column or (
            "title"
            if isinstance(first_example.get("title"), str)
            and str(first_example.get("title")).strip()
            else None
        )
    else:
        token_tag_like = _is_token_tag_example(first_example)
        if token_tag_like:
            resolved_text_column = text_column or "tokens"
            resolved_title_column = title_column or None
            features_obj = getattr(dataset, "features", None)
            ner_feature = None
            if features_obj is not None:
                if hasattr(features_obj, "get"):
                    ner_feature = features_obj.get("ner_tags")
                else:
                    try:
                        ner_feature = features_obj["ner_tags"]
                    except Exception:  # pragma: no cover - defensive
                        ner_feature = None
            token_tag_label_lookup, token_tag_label_names = _token_tag_label_lookup(
                ner_feature
            )
        else:
            resolved_text_column = text_column or _guess_text_column(first_example)
            resolved_title_column = title_column or _guess_title_column(
                first_example, resolved_text_column
            )

    rows: list[dict[str, Any]] = []
    text_lengths: list[int] = []
    ground_truth_entities: dict[str, set[str]] = {}
    ground_truth_counts: list[int] = []
    documents_with_ground_truth_entities = 0

    limit = len(dataset)
    if max_documents is not None:
        limit = min(limit, max_documents)
    if debug_document_limit is not None:
        if debug_document_limit <= 0:
            raise ValueError("debug_document_limit must be a positive integer")
        limit = min(limit, debug_document_limit)
        logger.info(
            "Debug document limit applied: using %s documents (requested limit: %s)",
            limit,
            debug_document_limit,
        )
    for idx in range(limit):
        example = dataset[idx]

        metadata_values: dict[str, Any] = {}

        if docred_like:
            processed = _preprocess_docred_like_example(example, f"Document {idx}")
            text = processed["text"] or ""
            title = processed["title"]
            if processed["metadata"]:
                metadata_values.update(processed["metadata"])
            parsed_entities = {
                entity for entity in processed.get("entities", []) if entity
            }
        elif token_tag_like:
            processed = _preprocess_token_tag_example(
                example, f"Document {idx}", token_tag_label_lookup
            )
            text = processed["text"] or ""
            title = processed["title"]
            if processed["metadata"]:
                metadata_values.update(processed["metadata"])
            parsed_entities = {
                entity for entity in processed.get("entities", []) if entity
            }
        else:
            text = str(example[resolved_text_column])
            title = (
                str(example[resolved_title_column])
                if resolved_title_column is not None
                else f"Document {idx}"
            )
            parsed_entities = set()

        if metadata_columns:
            for key in metadata_columns:
                if key in example:
                    metadata_values[key] = example.get(key)

        metadata = metadata_values or None

        text_lengths.append(len(text))

        document_id = example.get("id", idx)
        rows.append(
            {
                "id": document_id,
                "title": title,
                "text": text,
                "metadata": metadata,
                "creation_date": _normalise_creation_date(None),
            }
        )

        if docred_like:
            if parsed_entities:
                documents_with_ground_truth_entities += 1
            ground_truth_counts.append(len(parsed_entities))
            ground_truth_entities[_stringify_identifier(document_id) or str(idx)] = parsed_entities
        elif token_tag_like:
            if parsed_entities:
                documents_with_ground_truth_entities += 1
            ground_truth_counts.append(len(parsed_entities))
            ground_truth_entities[
                _stringify_identifier(document_id) or str(idx)
            ] = parsed_entities
        elif entity_column:
            raw_entities = example.get(entity_column)
            parsed_from_column = {
                value
                for value in _stringify_collection(raw_entities)
                if value is not None
            }
            if parsed_from_column:
                documents_with_ground_truth_entities += 1
            ground_truth_counts.append(len(parsed_from_column))
            ground_truth_entities[
                _stringify_identifier(document_id) or str(idx)
            ] = parsed_from_column

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
    if docred_like:
        dataset_summary["docred_like_format"] = True
    if token_tag_like:
        token_tag_info: dict[str, Any] = {
            "token_column": resolved_text_column,
            "tag_column": "ner_tags",
        }
        if token_tag_label_names:
            token_tag_info["label_count"] = int(len(token_tag_label_names))
            token_tag_info["label_examples"] = token_tag_label_names[:10]
        dataset_summary["token_tag_format"] = token_tag_info
    if metadata_columns:
        dataset_summary["metadata_columns"] = metadata_columns
    if debug_document_limit is not None:
        dataset_summary["debug_document_limit"] = debug_document_limit
    if ground_truth_counts:
        total_documents = len(ground_truth_counts) or 1
        ground_truth_summary: dict[str, Any] = {
            "documents_with_annotations": int(documents_with_ground_truth_entities),
            "average_entities_per_document": (
                float(sum(ground_truth_counts) / total_documents)
                if ground_truth_counts
                else 0.0
            ),
            "unique_entity_count": int(
                len({
                    entity
                    for values in ground_truth_entities.values()
                    for entity in values
                })
            ),
        }
        if docred_like and not entity_column:
            ground_truth_summary["source"] = "vertexSet"
        elif token_tag_like and not entity_column:
            ground_truth_summary["source"] = "ner_tags"
        elif entity_column:
            ground_truth_summary["column"] = entity_column
        dataset_summary["ground_truth_entities"] = ground_truth_summary

    return documents, dataset_summary, ground_truth_entities


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
    file_path = output_dir / filename
    if not file_path.exists():
        return None
    return pd.read_parquet(file_path)


def _normalize_token(value: Any) -> Optional[str]:
    """Normalise identifiers for comparison."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text.lower() if text else None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return None
        return str(value)
    return str(value).strip().lower() or None


def _has_content(value: Any) -> bool:
    """Return True when the provided value contains semantic content."""

    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set)):
        return any(_has_content(item) for item in value)
    if isinstance(value, dict):
        return any(_has_content(item) for item in value.values())
    if isinstance(value, (float, int, np.floating, np.integer)):
        return not pd.isna(value)
    return True


def _count_items(value: Any) -> int:
    """Estimate the number of items represented by a value."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, dict):
        return len(value)
    return 1


def _stringify_identifier(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        for key in ("name", "title", "entity", "value", "text"):
            if key in value:
                nested = _stringify_identifier(value[key])
                if nested is not None:
                    return nested
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return None
        return str(value)
    text = str(value).strip()
    return text or None


def _stringify_collection(value: Any) -> list[str | None]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    elif isinstance(value, set):
        items = list(value)
    elif isinstance(value, np.ndarray):
        items = value.tolist()
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    parsed = None
            if isinstance(parsed, (list, tuple, set, np.ndarray)):
                items = list(parsed)
            else:
                items = [text]
        else:
            items = [text]
    else:
        items = [value]

    result: list[str | None] = []
    for item in items:
        result.append(_stringify_identifier(item))
    return result


def _entity_identifier_sets(entities: pd.DataFrame) -> dict[str, set[str]]:
    identifiers: dict[str, set[str]] = {}
    for column in ["title", "id", "human_readable_id"]:
        if column in entities.columns:
            normalised = {
                token
                for token in (_normalize_token(value) for value in entities[column])
                if token is not None
            }
            if normalised:
                identifiers[column] = normalised
    return identifiers


def _entities_by_document(
    entities: Optional[pd.DataFrame],
    text_units: Optional[pd.DataFrame],
) -> dict[str, set[str]]:
    if entities is None or entities.empty or text_units is None or text_units.empty:
        return {}

    if "text_unit_ids" not in entities.columns:
        return {}

    text_unit_id_column = "id" if "id" in text_units.columns else None
    if text_unit_id_column is None:
        return {}

    document_column = None
    for candidate in ["document_ids", "document_id", "document"]:
        if candidate in text_units.columns:
            document_column = candidate
            break
    if document_column is None:
        return {}

    text_unit_to_documents: dict[str, set[str]] = {}
    for _, row in text_units.iterrows():
        text_unit_identifier = _stringify_identifier(row.get(text_unit_id_column))
        if text_unit_identifier is None:
            continue
        raw_doc_ids = _stringify_collection(row.get(document_column))
        doc_ids = {
            doc_id
            for doc_id in raw_doc_ids
            if doc_id is not None
        }
        if doc_ids:
            text_unit_to_documents[text_unit_identifier] = doc_ids

    document_entities: dict[str, set[str]] = defaultdict(set)
    for _, row in entities.iterrows():
        entity_title = row.get("title")
        if not _has_content(entity_title):
            continue
        entity_name = str(entity_title).strip()
        for text_unit_id in _stringify_collection(row.get("text_unit_ids")):
            if text_unit_id is None:
                continue
            for document_id in text_unit_to_documents.get(text_unit_id, set()):
                document_entities[document_id].add(entity_name)

    return {doc_id: set(values) for doc_id, values in document_entities.items()}


def _normalised_lookup(values: Iterable[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for value in values:
        if value is None:
            continue
        token = _normalize_token(value)
        if token is None:
            continue
        lookup.setdefault(token, value)
    return lookup


def _safe_divide(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return float(numerator / denominator)


def _evaluate_ner(
    ground_truth_entities: dict[str, set[str]],
    predicted_entities: dict[str, set[str]],
    example_limit: int = 5,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "status": "missing_ground_truth" if not ground_truth_entities else "ok",
        "documents_with_ground_truth": int(len(ground_truth_entities)),
        "documents_with_predictions": 0,
        "documents_missing_predictions": 0,
        "documents_with_annotations": 0,
        "documents_with_exact_match": 0,
        "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "macro": {"precision": None, "recall": None, "f1": None},
    }

    if not ground_truth_entities:
        return metrics

    total_tp = total_fp = total_fn = 0
    doc_precisions: list[float] = []
    doc_recalls: list[float] = []
    doc_f1s: list[float] = []
    false_negative_examples: list[dict[str, Any]] = []
    false_positive_examples: list[dict[str, Any]] = []

    docs_with_predictions = 0
    docs_with_exact_match = 0
    docs_with_annotations = 0

    extra_predicted_documents = [
        doc_id
        for doc_id in predicted_entities
        if doc_id not in ground_truth_entities
    ]

    for document_id, truth_values in ground_truth_entities.items():
        truth_lookup = _normalised_lookup(truth_values)
        pred_lookup = _normalised_lookup(predicted_entities.get(document_id, set()))
        truth_norm = set(truth_lookup)
        pred_norm = set(pred_lookup)

        if truth_norm:
            docs_with_annotations += 1

        if pred_norm:
            docs_with_predictions += 1
        elif truth_norm:
            metrics["documents_missing_predictions"] += 1

        tp = len(truth_norm & pred_norm)
        fp = len(pred_norm - truth_norm)
        fn = len(truth_norm - pred_norm)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = (
            _safe_divide(tp, tp + fp)
            if pred_norm
            else (1.0 if not truth_norm else 0.0)
        )
        recall = (
            _safe_divide(tp, tp + fn)
            if truth_norm
            else (1.0 if not pred_norm else 0.0)
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        doc_precisions.append(precision)
        doc_recalls.append(recall)
        doc_f1s.append(f1)

        if fn:
            false_negative_examples.append(
                {
                    "document_id": document_id,
                    "missing": sorted(truth_lookup[token] for token in truth_norm - pred_norm),
                }
            )
        if fp:
            false_positive_examples.append(
                {
                    "document_id": document_id,
                    "extra": sorted(pred_lookup[token] for token in pred_norm - truth_norm),
                }
            )

        if truth_norm == pred_norm:
            docs_with_exact_match += 1

    metrics["documents_with_predictions"] = int(docs_with_predictions)
    metrics["documents_with_exact_match"] = int(docs_with_exact_match)
    metrics["documents_with_annotations"] = int(docs_with_annotations)

    precision_micro = _safe_divide(total_tp, total_tp + total_fp)
    recall_micro = _safe_divide(total_tp, total_tp + total_fn)
    f1_micro = (
        2 * precision_micro * recall_micro / (precision_micro + recall_micro)
        if (precision_micro + recall_micro) > 0
        else 0.0
    )

    metrics["micro"] = {
        "precision": precision_micro,
        "recall": recall_micro,
        "f1": f1_micro,
    }

    if doc_precisions:
        metrics["macro"] = {
            "precision": float(sum(doc_precisions) / len(doc_precisions)),
            "recall": float(sum(doc_recalls) / len(doc_recalls)),
            "f1": float(sum(doc_f1s) / len(doc_f1s)),
        }

    metrics["examples"] = {
        "false_negatives": false_negative_examples[:example_limit],
        "false_positives": false_positive_examples[:example_limit],
    }

    if extra_predicted_documents:
        metrics["extra_predicted_documents"] = extra_predicted_documents[:example_limit]

    return metrics


def _evaluate_entities(
    entities: Optional[pd.DataFrame],
    relationships: Optional[pd.DataFrame],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "total_entities": 0,
        "entities_with_descriptions": 0,
        "entities_with_relationships": 0,
        "isolated_entities": 0,
        "identifier_counts": {},
        "average_mentions_per_entity": 0.0,
    }

    if entities is None or entities.empty:
        metrics["status"] = "missing"
        return metrics

    metrics["total_entities"] = int(len(entities))

    identifier_sets = _entity_identifier_sets(entities)
    metrics["identifier_counts"] = {
        key: len(value) for key, value in identifier_sets.items()
    }

    if "title" in identifier_sets:
        non_null_titles = entities["title"].apply(_normalize_token).dropna()
        metrics["duplicate_titles"] = int(
            len(non_null_titles) - len(identifier_sets["title"])
        )
    else:
        metrics["duplicate_titles"] = None

    if "description" in entities.columns:
        metrics["entities_with_descriptions"] = int(
            entities["description"].apply(_has_content).sum()
        )

    mention_counts: list[int] = []
    if "text_unit_ids" in entities.columns:
        mention_counts = [
            _count_items(value) for value in entities["text_unit_ids"]
        ]
    elif "degree" in entities.columns:
        # some pipelines expose degree as the number of incident relationships
        mention_counts = [
            value if isinstance(value, (int, float, np.integer, np.floating)) else 0
            for value in entities["degree"]
        ]

    if mention_counts:
        metrics["average_mentions_per_entity"] = float(
            sum(mention_counts) / max(len(mention_counts), 1)
        )

    connected_entities: set[str] = set()
    if relationships is not None and not relationships.empty:
        for column in ["source", "target"]:
            if column in relationships.columns:
                connected_entities.update(
                    token
                    for token in (
                        _normalize_token(value) for value in relationships[column]
                    )
                    if token is not None
                )

    title_set = identifier_sets.get("title", set())
    entities_with_edges = connected_entities & title_set if title_set else connected_entities
    metrics["entities_with_relationships"] = int(len(entities_with_edges))
    metrics["isolated_entities"] = int(
        metrics["total_entities"] - metrics["entities_with_relationships"]
    )
    metrics["share_with_relationships"] = (
        float(metrics["entities_with_relationships"] / metrics["total_entities"])
        if metrics["total_entities"]
        else 0.0
    )

    return metrics


def _evaluate_relationships(
    relationships: Optional[pd.DataFrame],
    entity_identifiers: dict[str, set[str]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "total_relationships": 0,
        "relationships_with_descriptions": 0,
        "relationships_with_text_units": 0,
        "self_loops": 0,
        "unique_pairs": 0,
        "invalid_source_references": 0,
        "invalid_target_references": 0,
        "weight_summary": None,
    }

    if relationships is None or relationships.empty:
        metrics["status"] = "missing"
        return metrics

    metrics["total_relationships"] = int(len(relationships))

    def _is_valid_reference(value: Any) -> bool:
        token = _normalize_token(value)
        if token is None:
            return False
        return any(token in tokens for tokens in entity_identifiers.values())

    sources = (
        relationships["source"].apply(_normalize_token)
        if "source" in relationships.columns
        else pd.Series(dtype="object")
    )
    targets = (
        relationships["target"].apply(_normalize_token)
        if "target" in relationships.columns
        else pd.Series(dtype="object")
    )

    if not sources.empty and not targets.empty:
        pair_index = pd.MultiIndex.from_arrays([sources, targets])
        metrics["unique_pairs"] = int(pair_index.nunique())
        metrics["self_loops"] = int((sources == targets).sum())

    if "description" in relationships.columns:
        metrics["relationships_with_descriptions"] = int(
            relationships["description"].apply(_has_content).sum()
        )

    if "text_unit_ids" in relationships.columns:
        text_unit_counts = [
            _count_items(value) for value in relationships["text_unit_ids"]
        ]
        metrics["relationships_with_text_units"] = int(
            sum(1 for count in text_unit_counts if count > 0)
        )
        if text_unit_counts:
            metrics["average_text_units_per_relationship"] = float(
                sum(text_unit_counts) / max(len(text_unit_counts), 1)
            )

    if "weight" in relationships.columns:
        weights = pd.to_numeric(relationships["weight"], errors="coerce").dropna()
        if not weights.empty:
            metrics["weight_summary"] = {
                "min": float(weights.min()),
                "max": float(weights.max()),
                "mean": float(weights.mean()),
            }

    source_valid: list[bool] = []
    target_valid: list[bool] = []
    if "source" in relationships.columns:
        source_valid = [
            _is_valid_reference(value) for value in relationships["source"]
        ]
        metrics["invalid_source_references"] = int(
            sum(1 for valid in source_valid if not valid)
        )
    if "target" in relationships.columns:
        target_valid = [
            _is_valid_reference(value) for value in relationships["target"]
        ]
        metrics["invalid_target_references"] = int(
            sum(1 for valid in target_valid if not valid)
        )

    valid_pairs = 0
    if source_valid and target_valid:
        valid_pairs = sum(
            1 for s_valid, t_valid in zip(source_valid, target_valid) if s_valid and t_valid
        )
    metrics["valid_relationship_pairs"] = int(valid_pairs)
    metrics["share_with_valid_references"] = (
        float(valid_pairs / metrics["total_relationships"])
        if metrics["total_relationships"]
        else 0.0
    )

    return metrics


def _graph_summary(artifacts: dict[str, pd.DataFrame], top_k: int) -> dict[str, Any]:
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
            c
            for c in ["source", "target", "description", "weight"]
            if c in relationships.columns
        ]
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
) -> dict[str, Any]:
    workflows = stats_data.get("workflows", {}) if stats_data else {}
    return {
        "wall_time_seconds": float(wall_time_seconds),
        "reported_total_runtime_seconds": float(stats_data.get("total_runtime", 0.0)),
        "input_load_time_seconds": float(stats_data.get("input_load_time", 0.0)),
        "workflow_durations_seconds": workflows,
    }


async def _run_indexing_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    root_dir = args.workspace_dir.resolve()
    _ensure_workspace(root_dir)
    run_logger = _setup_logger(root_dir / "logs")
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

    indexing_method = IndexingMethod(args.indexing_method)
    run_logger.info("Indexing method: %s", indexing_method)

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

    indexing_metrics = _evaluate_indexing_run(stats_data, wall_time_seconds)

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


if __name__ == "__main__":
    main()
