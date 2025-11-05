"""Shared evaluation and dataset helpers for finance scripts."""

from __future__ import annotations

import ast
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset


logger = logging.getLogger(__name__)


def _guess_text_column(example: dict[str, Any]) -> str:
    """Return a best-effort guess for the primary text column."""

    preferred = [
        "text",
        "content",
        "article",
        "document",
        "body",
        "all_text",
        "summary",
    ]

    for key in preferred:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return key

    for key, value in example.items():
        if key.lower() == "title":
            continue
        if isinstance(value, str) and value.strip():
            return key

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
        '"',
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
        else:
            title = default_title

    raw_entities = example.get("entities")
    metadata: dict[str, Any] = {}
    if isinstance(raw_entities, dict):
        metadata["entity_metadata"] = raw_entities

    raw_tags = example.get("ner_tags")
    entities, span_count, label_counts = _extract_token_tag_entities(
        tokens, raw_tags, label_lookup
    )

    if span_count:
        metadata["entity_spans"] = span_count
    if label_counts:
        metadata["entity_label_counts"] = label_counts

    return {
        "title": title,
        "text": text,
        "entities": entities,
        "metadata": metadata or None,
    }


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


def _load_finance_documents(
    dataset_name: str,
    split: str,
    max_documents: Optional[int],
    debug_document_limit: Optional[int],
    text_column: Optional[str],
    title_column: Optional[str],
    metadata_columns: Optional[list[str]],
    entity_column: Optional[str] = None,
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
            ground_truth_entities[_stringify_identifier(document_id) or str(idx)] = (
                parsed_entities
            )
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


def _has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return not pd.isna(value)
    return True


def _count_items(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, np.ndarray):
        return int(value.size)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    parsed = None
            if isinstance(parsed, (list, tuple, set, np.ndarray)):
                return len(parsed)
        return 1
    return 1


def _normalize_token(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text.casefold() if text else None


def _entity_identifier_sets(entities: pd.DataFrame) -> dict[str, set[str]]:
    columns = [
        column
        for column in ["title", "description", "source", "id"]
        if column in entities.columns
    ]
    identifier_sets: dict[str, set[str]] = {column: set() for column in columns}

    for _, row in entities.iterrows():
        for column in columns:
            value = row.get(column)
            token = _normalize_token(value)
            if token is not None:
                identifier_sets[column].add(token)

    return identifier_sets


def _normalised_lookup(values: Iterable[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for value in values:
        token = _normalize_token(value)
        if token is not None:
            lookup[token] = str(value).strip()
    return lookup


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


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


def _entities_by_document(
    entities: Optional[pd.DataFrame],
    text_units: Optional[pd.DataFrame],
) -> dict[str, set[str]]:
    if entities is None or entities.empty:
        return {}
    if text_units is None or text_units.empty:
        return {}

    text_unit_id_column = None
    for candidate in ["id", "text_unit_id", "text_unit_ids"]:
        if candidate in text_units.columns:
            text_unit_id_column = candidate
            break
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
                    {
                        _normalize_token(value)
                        for value in relationships[column]
                        if _normalize_token(value) is not None
                    }
                )

    if connected_entities:
        metrics["entities_with_relationships"] = int(len(connected_entities))
        metrics["isolated_entities"] = int(
            len(identifier_sets.get("title", set()) - connected_entities)
        )
    else:
        metrics["entities_with_relationships"] = 0
        metrics["isolated_entities"] = metrics["total_entities"]

    return metrics


def _evaluate_relationships(
    relationships: Optional[pd.DataFrame],
    entity_identifiers: dict[str, set[str]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "total_relationships": 0,
        "with_description": 0,
        "with_weight": 0,
        "unresolved_entities": 0,
        "self_loops": 0,
    }

    if relationships is None or relationships.empty:
        metrics["status"] = "missing"
        return metrics

    metrics["total_relationships"] = int(len(relationships))

    if "description" in relationships.columns:
        metrics["with_description"] = int(
            relationships["description"].apply(_has_content).sum()
        )

    if "weight" in relationships.columns:
        metrics["with_weight"] = int(
            relationships["weight"].apply(
                lambda value: isinstance(value, (int, float, np.integer, np.floating))
                and not pd.isna(value)
            ).sum()
        )

    unresolved_entities = 0
    self_loops = 0

    normalised_titles = entity_identifiers.get("title", set())

    for _, row in relationships.iterrows():
        source = _normalize_token(row.get("source"))
        target = _normalize_token(row.get("target"))

        if source is None or target is None:
            unresolved_entities += 1
            continue

        if source == target:
            self_loops += 1

        if source not in normalised_titles or target not in normalised_titles:
            unresolved_entities += 1

    metrics["unresolved_entities"] = int(unresolved_entities)
    metrics["self_loops"] = int(self_loops)

    return metrics


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

