# Baseline consolidation and observability checklist

This document captures the current defaults, workflows, and instrumentation that
inform Phase 0 of the GraphRAG expansion plan. Use it as a living reference when
changing model providers or extending observability.

## Provider-agnostic default configuration

GraphRAG exposes a minimal set of environment variables that make the default
LLM and embedding selection provider-neutral without breaking existing OpenAI
setups:

| Environment variable | Purpose | Default |
| --- | --- | --- |
| `GRAPHRAG_DEFAULT_MODEL_PROVIDER` | Sets the provider prefix used by LiteLLM when no provider is specified in user configuration. | `openai` |
| `GRAPHRAG_DEFAULT_CHAT_MODEL` | Overrides the default chat model name that populates generated settings files and CLI defaults. | `gpt-4-turbo-preview` |
| `GRAPHRAG_DEFAULT_EMBEDDING_MODEL` | Overrides the default embedding model name used when none is specified. | `text-embedding-3-small` |

These variables only affect fallback defaults. Explicit configuration in
`settings.yaml` or environment-specific overrides take precedence. The values
are trimmed automatically so accidental whitespace in `.env` files does not
create malformed model identifiers.

## Indexing workflow inventory

The `PipelineFactory` composes several reusable stages that ingest content into
the knowledge graph. The standard pipeline registers the following sequence:

1. **Document ingestion** – `graphrag.index.workflows.load_input_documents`
   normalizes source files and persists a canonical documents table.
2. **Base text-unit creation** – `create_base_text_units` and
   `create_final_documents` transform documents into chunked text units with
   consistent metadata payloads.
3. **Entity and relationship extraction** – `extract_graph` and
   `finalize_graph` prompt the configured chat model to identify entities and
   their relationships, emitting node and edge tables.
4. **Covariate extraction** – `extract_covariates` pulls supplementary
   structured signals (e.g., financial indicators) that can augment downstream
   analysis.
5. **Community detection and reporting** – `create_communities`,
   `create_final_text_units`, and `create_community_reports` cluster related
   entities and generate natural-language summaries.
6. **Embedding generation** – `generate_text_embeddings` projects text units
   into the configured vector store for retrieval.

The incremental update workflows reuse the same components but swap in the
`load_update_documents` entrypoint and append stages such as
`update_entities_relationships`, `update_covariates`, and
`update_text_embeddings` to merge deltas into the canonical graph. Temporal
metadata on documents and text units remains intact during chunking, enabling
time-sliced analysis without full rebuilds.

## Query workflow inventory

GraphRAG provides three primary query orchestrators surfaced by
`graphrag.query.factory`: local search, global search, and DRIFT search. Each
workflow layers the same components differently:

- **Local search** expands around the most similar text units and entities,
  building a localized answer synthesis prompt.
- **Global search** stitches together community-level summaries, allowing long
  horizon reasoning over the entire graph.
- **DRIFT search** detects topical changes by comparing multiple prompt
  perspectives and following up with local drills.

All query paths accept configurable language model IDs, making the new provider
controls usable for both indexing and question answering.

## LLM observability hooks

The LiteLLM request wrappers now emit structured log events for every chat or
embedding invocation:

- `llm_request_start` records provider, model, and request identifiers (when
  supplied) before the upstream call.
- `llm_request_success` includes the same identifiers plus the end-to-end latency
  in milliseconds for successful calls.
- Failure paths attach identical metadata to the exception log entry to simplify
  correlation in log aggregators.

Downstream tooling (e.g., OpenTelemetry processors or simple log forwarders)
can collect these events to build dashboards comparing providers, deployments,
and workloads during the upcoming Hugging Face integration experiments.
