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
from datasets import Dataset, load_dataset

from graphrag.api.index import build_index
from graphrag.config.create_graphrag_config import create_graphrag_config
from graphrag.config.embeddings import default_embeddings
from graphrag.config.enums import IndexingMethod
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.workflows.factory import PipelineFactory
from graphrag.language_model.providers.huggingface.chat_model import (
    HuggingFaceChatModel,
)


def _is_token_tag_example(example: dict[str, Any]) -> bool:
    """Identify token-level NER examples with parallel token/tag sequences."""

    tokens = example.get("tokens")
    tags = example.get("ner_tags")

    if not isinstance(tokens, (list, tuple)) or not isinstance(tags, (list, tuple)):
        return False

    if not tokens or not tags or len(tokens) != len(tags):
        return False

    return True
    
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
    dataset = _load_dataset_from_source(dataset_name, split)
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
        # logger.info(
        #     "Debug document limit applied: using %s documents (requested limit: %s)",
        #     limit,
        #     debug_document_limit,
        # )
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
        # logger.info(f'parsed_entities:{parsed_entities}')
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

    def _sorted_vertices(entity_map: dict[str, set[str]]) -> list[str]:
        aggregated: dict[str, str] = {}
        for values in entity_map.values():
            for value in values:
                token = _normalize_token(value)
                if token is None:
                    continue
                aggregated.setdefault(token, str(value).strip())
        return sorted(aggregated.values(), key=lambda item: item.casefold())

    if not ground_truth_entities:
        metrics["sorted_ground_truth_vertices"] = []
        metrics["sorted_extracted_truth_vertices"] = _sorted_vertices(predicted_entities)
        return metrics

    metrics["sorted_ground_truth_vertices"] = _sorted_vertices(ground_truth_entities)
    metrics["sorted_extracted_truth_vertices"] = _sorted_vertices(predicted_entities)

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

def _load_dataset_from_source(dataset_name: str, split: str) -> Dataset:
    """Load a dataset from Hugging Face or a local file."""

    candidate_path = Path(dataset_name)
    if candidate_path.exists():
        suffix = candidate_path.suffix.lower()
        if suffix == ".csv":
            dataframe = pd.read_csv(candidate_path)
        elif suffix in {".json", ".jsonl"}:
            dataframe = pd.read_json(candidate_path, lines=suffix == ".jsonl")
        else:
            raise ValueError(
                "Unsupported local dataset format. "
                "Provide a .csv, .json, or .jsonl file or a Hugging Face dataset name."
            )

        if dataframe.empty:
            raise ValueError("Loaded dataset is empty.")

        return Dataset.from_pandas(dataframe.reset_index(drop=True), preserve_index=False)

    return load_dataset(dataset_name, split=split)

def _normalise_creation_date(value: Any) -> str:
    """Convert a raw creation date value into an ISO-8601 string."""

    if isinstance(value, str):
        value = value.strip() or None

    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if timestamp is not None and not pd.isna(timestamp):
        return timestamp.isoformat()

    return pd.Timestamp.now(tz="UTC").isoformat()

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