"""HippoRAG-backed graph extraction workflow."""

from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict
from typing import Any, Iterable, Mapping

import pandas as pd

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.misc_utils import compute_mdhash_id

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage

from .extract_graph import get_summarized_entities_relationships

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Execute the HippoRAG graph extraction workflow."""

    logger.info("Workflow started: extract_graph_hipporag")
    text_units = await load_table_from_storage("text_units", context.output_storage)

    entities, relationships, raw_entities, raw_relationships = await extract_graph(
        config=config,
        text_units=text_units,
        callbacks=context.callbacks,
        cache=context.cache,
    )

    await write_table_to_storage(entities, "entities", context.output_storage)
    await write_table_to_storage(relationships, "relationships", context.output_storage)

    if config.snapshots.raw_graph:
        await write_table_to_storage(raw_entities, "raw_entities", context.output_storage)
        await write_table_to_storage(
            raw_relationships, "raw_relationships", context.output_storage
        )

    logger.info("Workflow completed: extract_graph_hipporag")
    return WorkflowFunctionOutput(result={"entities": entities, "relationships": relationships})


async def extract_graph(
    config: GraphRagConfig,
    text_units: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run HippoRAG indexing and adapt its outputs to the GraphRAG schema."""

    if text_units is None or text_units.empty:
        msg = "No text units available for HippoRAG indexing."
        raise ValueError(msg)

    extract_llm_settings = config.get_language_model_config(config.extract_graph.model_id)
    embedding_settings = config.get_language_model_config(config.embed_text.model_id)
    summarization_settings = config.get_language_model_config(
        config.summarize_descriptions.model_id
    )
    summarization_strategy = config.summarize_descriptions.resolved_strategy(
        config.root_dir, summarization_settings
    )

    hippo_config = _build_hipporag_config(
        config=config,
        llm_config=extract_llm_settings,
        embedding_config=embedding_settings,
    )

    if "text" not in text_units.columns:
        msg = "Text units table must include a 'text' column for HippoRAG indexing."
        raise ValueError(msg)

    if "id" not in text_units.columns:
        text_units = text_units.reset_index().rename(columns={"index": "id"})

    docs = text_units["text"].astype(str).tolist()
    text_unit_ids = text_units["id"].astype(str).tolist()
    chunk_to_text_units = _map_chunk_ids_to_text_units(docs, text_unit_ids)

    hippo_entities, hippo_relationships = await asyncio.to_thread(
        _run_hipporag_index,
        hippo_config,
        docs,
        chunk_to_text_units,
    )

    raw_entities = hippo_entities.copy(deep=True)
    raw_relationships = hippo_relationships.copy(deep=True)

    if hippo_entities.empty or hippo_relationships.empty:
        msg = "HippoRAG extraction yielded no entities or relationships."
        logger.error(msg)
        raise ValueError(msg)

    entities, relationships = await get_summarized_entities_relationships(
        extracted_entities=hippo_entities,
        extracted_relationships=hippo_relationships,
        callbacks=callbacks,
        cache=cache,
        summarization_strategy=summarization_strategy,
        summarization_num_threads=summarization_settings.concurrent_requests,
    )

    # Re-attach text unit references lost during summarisation.
    entities = _restore_text_unit_metadata(entities, raw_entities, "title")
    relationships = _restore_text_unit_metadata(
        relationships, raw_relationships, ("source", "target")
    )

    # Ensure sorted identifiers for downstream determinism.
    entities["text_unit_ids"] = entities["text_unit_ids"].apply(lambda ids: sorted(set(ids)))
    relationships["text_unit_ids"] = relationships["text_unit_ids"].apply(
        lambda ids: sorted(set(ids))
    )

    return entities, relationships, raw_entities, raw_relationships


def _run_hipporag_index(
    hippo_config: BaseConfig,
    docs: list[str],
    chunk_to_text_units: Mapping[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hippo = HippoRAG(global_config=hippo_config)
    logger.info(f'docs:\n\t{docs}')
    hippo.index(docs)

    openie_results, _ = hippo.load_existing_openie([])
    logger.info(f'openie_results:{openie_results}')
    logger.info(f'chunk_to_text_units:{chunk_to_text_units}')
    return _convert_openie_to_tables(openie_results, chunk_to_text_units)


def _build_hipporag_config(
    config: GraphRagConfig,
    llm_config: LanguageModelConfig,
    embedding_config: LanguageModelConfig,
) -> BaseConfig:
    save_dir = os.path.join(config.output.base_dir, "hipporag")
    os.makedirs(save_dir, exist_ok=True)

    llm_name, llm_kwargs = _map_llm_config(llm_config)
    embedding_name, embedding_kwargs = _map_embedding_config(embedding_config)

    temperature = getattr(llm_config, "temperature", None)
    max_tokens = getattr(llm_config, "max_tokens", None) or getattr(
        llm_config, "max_completion_tokens", None
    )

    base_kwargs: dict[str, Any] = {
        "save_dir": save_dir,
        # "llm_model_name": llm_name,
        "llm_name": llm_name,        
        "embedding_model_name": embedding_name,
        "temperature": temperature,
        "max_new_tokens": max_tokens,
        "max_retry_attempts": llm_config.max_retries,
    }
    base_kwargs.update(llm_kwargs)
    base_kwargs.update(embedding_kwargs)

    # Remove None values so BaseConfig keeps defaults where appropriate.
    clean_kwargs = {k: v for k, v in base_kwargs.items() if v is not None}
    hippo_config = BaseConfig(**clean_kwargs)

    _maybe_set_hf_token(llm_config)
    _maybe_set_hf_token(embedding_config)

    return hippo_config


def _map_llm_config(
    llm_config: LanguageModelConfig,
) -> tuple[str, Mapping[str, Any]]:
    params: dict[str, Any] = {}
    if llm_config.type in {"huggingface_chat", "huggingface"} or (
        llm_config.model_provider or ""
    ).lower() == "huggingface":
        llm_name = f"Transformers/{llm_config.model}"
    else:
        llm_name = llm_config.model

    provider = (llm_config.model_provider or "").lower()
    if provider == "azure":
        params["azure_endpoint"] = llm_config.api_base
    elif llm_config.api_base:
        params["llm_base_url"] = llm_config.api_base

    return llm_name, params


def _map_embedding_config(
    embedding_config: LanguageModelConfig,
) -> tuple[str, Mapping[str, Any]]:
    params: dict[str, Any] = {}
    provider = (embedding_config.model_provider or "").lower()
    if embedding_config.type in {"huggingface_embedding", "huggingface"} or provider == "huggingface":
        emb_name = f"Transformers/{embedding_config.model}"
    else:
        emb_name = embedding_config.model

    if provider == "azure":
        params["azure_embedding_endpoint"] = embedding_config.api_base
    elif embedding_config.api_base:
        params["embedding_base_url"] = embedding_config.api_base

    return emb_name, params


def _maybe_set_hf_token(model_config: LanguageModelConfig) -> None:
    token = (model_config.api_key or "").strip()
    if not token:
        return
    for env_var in ("HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN"):
        os.environ.setdefault(env_var, token)


def _map_chunk_ids_to_text_units(
    docs: Iterable[str], text_unit_ids: Iterable[str]
) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for doc, text_unit_id in zip(docs, text_unit_ids):
        chunk_id = compute_mdhash_id(doc, prefix="chunk-")
        mapping[chunk_id].add(text_unit_id)
    return mapping


def _convert_openie_to_tables(
    openie_results: list[dict[str, Any]],
    chunk_to_text_units: Mapping[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    entity_text_units: dict[str, set[str]] = defaultdict(set)
    entity_descriptions: dict[str, list[str]] = defaultdict(list)
    relationship_text_units: dict[tuple[str, str], set[str]] = defaultdict(set)
    relationship_descriptions: dict[tuple[str, str], list[str]] = defaultdict(list)
    relationship_weights: dict[tuple[str, str], float] = defaultdict(float)

    for doc in openie_results:
        chunk_id = doc.get("idx")
        passage = str(doc.get("passage", "")).strip()
        text_unit_ids = chunk_to_text_units.get(chunk_id, set())

        extracted_entities = doc.get("extracted_entities") or []
        extracted_triples = doc.get("extracted_triples") or []

        for entity in extracted_entities:
            entity_name = str(entity).strip()
            if not entity_name:
                continue
            entity_text_units[entity_name].update(text_unit_ids)
            if passage:
                entity_descriptions[entity_name].append(
                    f"{entity_name} mentioned in: {passage}"
                )

        for triple in extracted_triples:
            if len(triple) != 3:
                continue
            subject = str(triple[0]).strip()
            relation = str(triple[1]).strip()
            obj = str(triple[2]).strip()
            if not subject or not obj:
                continue

            summary = f"{subject} {relation} {obj}".strip()
            if not summary:
                summary = passage

            entity_text_units[subject].update(text_unit_ids)
            entity_text_units[obj].update(text_unit_ids)
            if summary:
                entity_descriptions[subject].append(summary)
                entity_descriptions[obj].append(summary)

            edge_key = (subject, obj)
            relationship_text_units[edge_key].update(text_unit_ids)
            if summary:
                relationship_descriptions[edge_key].append(summary)
            relationship_weights[edge_key] += 1.0

    entities_df = _build_entities_dataframe(entity_text_units, entity_descriptions)
    relationships_df = _build_relationships_dataframe(
        relationship_text_units, relationship_descriptions, relationship_weights
    )

    return entities_df, relationships_df


def _build_entities_dataframe(
    entity_text_units: Mapping[str, set[str]],
    entity_descriptions: Mapping[str, list[str]],
) -> pd.DataFrame:
    records = []
    for title, text_ids in entity_text_units.items():
        cleaned_title = title.strip()
        if not cleaned_title:
            continue
        descriptions = entity_descriptions.get(title, [])
        if not descriptions:
            continue
        records.append(
            {
                "title": cleaned_title,
                "type": "entity",
                "description": sorted(set(descriptions)),
                "text_unit_ids": sorted(set(text_ids)),
                "frequency": len(descriptions),
            }
        )

    if not records:
        return pd.DataFrame(
            columns=["title", "type", "description", "text_unit_ids", "frequency"]
        )

    return pd.DataFrame.from_records(records)


def _build_relationships_dataframe(
    relationship_text_units: Mapping[tuple[str, str], set[str]],
    relationship_descriptions: Mapping[tuple[str, str], list[str]],
    relationship_weights: Mapping[tuple[str, str], float],
) -> pd.DataFrame:
    records = []
    for (source, target), text_ids in relationship_text_units.items():
        src = source.strip()
        tgt = target.strip()
        if not src or not tgt:
            continue
        descriptions = relationship_descriptions.get((source, target), [])
        if not descriptions:
            continue
        weight = relationship_weights.get((source, target), 0.0)
        records.append(
            {
                "source": src,
                "target": tgt,
                "description": sorted(set(descriptions)),
                "text_unit_ids": sorted(set(text_ids)),
                "weight": float(weight),
            }
        )

    if not records:
        return pd.DataFrame(
            columns=["source", "target", "description", "text_unit_ids", "weight"]
        )

    return pd.DataFrame.from_records(records)


def _restore_text_unit_metadata(
    enriched_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    key_columns: str | tuple[str, str],
) -> pd.DataFrame:
    if isinstance(key_columns, str):
        join_keys = [key_columns]
    else:
        join_keys = list(key_columns)

    metadata_columns = [col for col in ["text_unit_ids", "frequency"] if col in raw_df.columns]
    if "weight" in raw_df.columns and "weight" not in metadata_columns:
        metadata_columns.append("weight")

    if not metadata_columns:
        return enriched_df

    merged = enriched_df.merge(
        raw_df.loc[:, [*join_keys, *metadata_columns]],
        on=join_keys,
        how="left",
        suffixes=("", "_raw"),
    )

    for column in metadata_columns:
        raw_col = f"{column}_raw"
        if raw_col in merged:
            merged[column] = merged[raw_col].combine_first(merged[column])
            merged.drop(columns=[raw_col], inplace=True)

    return merged

