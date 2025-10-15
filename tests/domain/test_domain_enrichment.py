import pandas as pd

from graphrag.config.create_graphrag_config import create_graphrag_config
from graphrag.domain.context import DomainContext
from graphrag.domain.enrichment import (
    annotate_covariates_with_entity_domains,
    apply_domain_rules,
    ensure_covariate_domain_columns,
    ensure_entity_domain_columns,
)
from graphrag.domain.rules import DomainEntityRule
from tests.verbs.util import DEFAULT_MODEL_CONFIG


def test_apply_domain_rules_sets_tags():
    df = pd.DataFrame(
        {
            "id": ["1"],
            "human_readable_id": [0],
            "title": ["Copper Futures"],
            "type": ["commodity"],
            "description": ["Copper supply outlook"],
            "text_unit_ids": [["tu-1"]],
            "frequency": [1],
            "degree": [2],
            "x": [0.0],
            "y": [0.0],
        }
    )
    context = DomainContext(
        domain_name="Finance",
        entity_rules=(
            DomainEntityRule(tag="commodity", match_types={"commodity"}, priority=1),
        ),
    )

    enriched = ensure_entity_domain_columns(df.copy(), context)
    enriched = apply_domain_rules(enriched, context)

    assert enriched.loc[0, context.entity_tag_column] == ["commodity"]
    assert enriched.loc[0, context.entity_primary_column] == "commodity"
    assert enriched.loc[0, context.entity_profile_column] == "Finance"


def test_annotate_covariates_with_entity_domains_maps_subjects():
    covariates = pd.DataFrame(
        {
            "id": ["c1"],
            "human_readable_id": [0],
            "covariate_type": ["claim"],
            "type": ["SUPPLY"],
            "description": ["Mine outage"],
            "subject_id": ["Copper Futures"],
            "object_id": ["NONE"],
            "status": ["TRUE"],
            "start_date": ["2024-01-01"],
            "end_date": ["2024-01-31"],
            "source_text": [["Source"]],
            "text_unit_id": ["tu-1"],
        }
    )
    entities = pd.DataFrame(
        {
            "title": ["Copper Futures"],
            "domain_tags": [["commodity"]],
            "domain_primary_tag": ["commodity"],
        }
    )
    context = DomainContext(
        domain_name="Finance",
        entity_rules=(DomainEntityRule(tag="commodity", match_types={"commodity"}),),
    )

    covariates = ensure_covariate_domain_columns(covariates, context)
    annotated = annotate_covariates_with_entity_domains(covariates, entities, context)

    assert annotated.loc[0, context.covariate_subject_tags_column] == ["commodity"]
    assert annotated.loc[0, context.covariate_primary_tag_column] == "commodity"
    assert annotated.loc[0, context.covariate_profile_column] == "Finance"


def test_domain_config_finance_profile_resolves_defaults():
    config = create_graphrag_config(
        {
            "models": DEFAULT_MODEL_CONFIG,
            "domain_intelligence": {"enabled": True, "profile": "finance"},
        }
    )

    context = config.resolved_domain_context()
    assert context is not None
    assert context.domain_name == "Commodities and Futures"
    assert context.covariate_type == "finance_factor"
    assert context.has_entity_rules()
