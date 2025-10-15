"""Helpers to apply domain intelligence to dataframes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from .context import DomainContext
from .rules import DomainEntityRule


def ensure_entity_domain_columns(df: pd.DataFrame, context: DomainContext | None) -> pd.DataFrame:
    """Ensure the entity dataframe contains the domain columns expected downstream."""

    if context is None:
        columns = ("domain_profile", "domain_tags", "domain_primary_tag")
    else:
        columns = (
            context.entity_profile_column or "domain_profile",
            context.entity_tag_column,
            context.entity_primary_column or "domain_primary_tag",
        )
    for column in columns:
        if column and column not in df.columns:
            df[column] = None
    return df


def apply_domain_rules(entities: pd.DataFrame, context: DomainContext) -> pd.DataFrame:
    """Apply domain tagging rules to the entity dataframe."""

    if context is None or not context.has_entity_rules():
        return entities

    tag_column = context.entity_tag_column
    primary_column = context.entity_primary_column
    profile_column = context.entity_profile_column

    def _evaluate_row(row: pd.Series) -> tuple[list[str] | None, str | None]:
        matches: list[DomainEntityRule] = []
        for rule in context.iter_rules():
            if rule.matches(
                title=row.get("title", ""),
                entity_type=row.get("type"),
                description=row.get("description"),
            ):
                matches.append(rule)
        if not matches:
            return None, None
        tags = sorted({rule.tag for rule in matches})
        primary_rule = min(matches, key=lambda r: r.priority)
        return tags, primary_rule.tag

    evaluated = entities.apply(_evaluate_row, axis=1, result_type="expand")
    if not evaluated.empty:
        if tag_column:
            entities.loc[:, tag_column] = evaluated[0]
        if primary_column:
            entities.loc[:, primary_column] = evaluated[1]
    if profile_column:
        entities.loc[:, profile_column] = context.domain_name
    return entities


def ensure_covariate_domain_columns(
    covariates: pd.DataFrame, context: DomainContext | None
) -> pd.DataFrame:
    """Guarantee covariate outputs include domain metadata columns."""

    if context is None:
        columns = (
            "domain_profile",
            "subject_domain_tags",
            "subject_domain_primary_tag",
        )
    else:
        columns = (
            context.covariate_profile_column or "domain_profile",
            context.covariate_subject_tags_column,
            context.covariate_primary_tag_column or "subject_domain_primary_tag",
        )
    for column in columns:
        if column and column not in covariates.columns:
            covariates[column] = None
    return covariates


def annotate_covariates_with_entity_domains(
    covariates: pd.DataFrame,
    entities: pd.DataFrame,
    context: DomainContext,
) -> pd.DataFrame:
    """Project entity domain annotations onto covariates."""

    if context is None:
        return covariates

    tags_column = context.entity_tag_column
    primary_column = context.entity_primary_column
    covariate_tags_column = context.covariate_subject_tags_column
    covariate_primary_column = context.covariate_primary_tag_column
    covariate_profile_column = context.covariate_profile_column

    if tags_column not in entities.columns:
        return covariates

    entity_lookup = entities.set_index("title")
    tag_series = entity_lookup.get(tags_column)
    primary_series = entity_lookup.get(primary_column) if primary_column else None
    profile_value = context.domain_name if covariate_profile_column else None

    def _lookup(value: Any, series: pd.Series | None) -> Any:
        if series is None:
            return None
        try:
            return series.get(value)
        except KeyError:
            return None

    covariates.loc[:, covariate_tags_column] = covariates["subject_id"].apply(
        lambda subject: _normalize_tags(_lookup(subject, tag_series))
    )
    if covariate_primary_column:
        covariates.loc[:, covariate_primary_column] = covariates["subject_id"].apply(
            lambda subject: _lookup(subject, primary_series)
        )
    if covariate_profile_column:
        covariates.loc[:, covariate_profile_column] = profile_value

    return covariates


def _normalize_tags(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]
