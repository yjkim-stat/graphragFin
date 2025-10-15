"""Configuration model for domain-aware enrichment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from graphrag.config.defaults import graphrag_config_defaults
from graphrag.domain.context import DomainContext
from graphrag.domain.profiles import get_profile
from graphrag.domain.rules import DomainEntityRule


class DomainEntityRuleConfig(BaseModel):
    """Pydantic schema for custom domain rules."""

    tag: str = Field(description="Tag to assign when the rule matches.")
    match_types: list[str] | None = Field(
        default=None,
        description="Entity type labels that trigger the rule.",
    )
    keywords: list[str] | None = Field(
        default=None,
        description="Keywords to search for in the entity title or description.",
    )
    match_titles: list[str] | None = Field(
        default=None,
        description="Title fragments that should activate the rule.",
    )
    attribute_overrides: dict[str, Any] | None = Field(
        default=None,
        description="Reserved for future use to copy additional attributes.",
    )
    priority: int = Field(
        default=100,
        description="Lower values represent higher priority when selecting the primary tag.",
    )
    primary: bool = Field(
        default=False,
        description="Hint that the rule should often be treated as primary when matched.",
    )

    def to_rule(self) -> DomainEntityRule:
        """Convert configuration into runtime rule object."""

        data = self.model_dump()
        return DomainEntityRule.from_config(data)


class DomainIntelligenceConfig(BaseModel):
    """Domain-aware enrichment options."""

    enabled: bool = Field(
        default=graphrag_config_defaults.domain_intelligence.enabled,
        description="Master switch for domain intelligence enrichment.",
    )
    profile: str | None = Field(
        default=graphrag_config_defaults.domain_intelligence.profile,
        description="Optional built-in profile name (e.g. 'finance').",
    )
    domain: str | None = Field(
        default=graphrag_config_defaults.domain_intelligence.domain,
        description="Override the human-readable domain name used in prompts.",
    )
    entity_rules: list[DomainEntityRuleConfig] = Field(
        default_factory=list,
        description="Additional custom rules appended after profile rules.",
    )
    entity_tag_column: str = Field(
        default=graphrag_config_defaults.domain_intelligence.entity_tag_column,
        description="Column name used to store entity domain tags.",
    )
    entity_primary_column: str | None = Field(
        default=graphrag_config_defaults.domain_intelligence.entity_primary_column,
        description="Column storing the primary domain tag for the entity.",
    )
    entity_profile_column: str | None = Field(
        default=graphrag_config_defaults.domain_intelligence.entity_profile_column,
        description="Column storing the active domain profile label.",
    )
    covariate_type: str | None = Field(
        default=graphrag_config_defaults.domain_intelligence.covariate_type,
        description="Override the covariate_type value emitted during extraction.",
    )
    covariate_entity_types: list[str] | None = Field(
        default_factory=lambda: (
            graphrag_config_defaults.domain_intelligence.covariate_entity_types
            or []
        ),
        description="Restrict claim extraction to these entity types when provided.",
    )
    covariate_description: str | None = Field(
        default=graphrag_config_defaults.domain_intelligence.covariate_description,
        description="Domain specific claim description passed to the extractor.",
    )
    covariate_prompt: str | None = Field(
        default=graphrag_config_defaults.domain_intelligence.covariate_prompt,
        description="Optional prompt override used for claim extraction.",
    )
    covariate_subject_tags_column: str = Field(
        default=graphrag_config_defaults.domain_intelligence.covariate_subject_tags_column,
        description="Column storing the mapped subject domain tags for covariates.",
    )
    covariate_primary_tag_column: str | None = Field(
        default=graphrag_config_defaults.domain_intelligence.covariate_primary_tag_column,
        description="Column storing the primary subject tag for covariates.",
    )
    covariate_profile_column: str | None = Field(
        default=graphrag_config_defaults.domain_intelligence.covariate_profile_column,
        description="Column storing the domain profile for covariates.",
    )
    community_graph_prompt: str | None = Field(
        default=graphrag_config_defaults.domain_intelligence.community_graph_prompt,
        description="Optional override for the graph-based community summary prompt.",
    )
    community_text_prompt: str | None = Field(
        default=graphrag_config_defaults.domain_intelligence.community_text_prompt,
        description="Optional override for the text-based community summary prompt.",
    )

    def resolved(self, root_dir: str) -> DomainContext | None:
        """Resolve the runtime domain context or return None when disabled."""

        if not self.enabled:
            return None

        profile = get_profile(self.profile)
        profile_context = profile.to_context() if profile else None

        domain_name = self.domain or (profile_context.domain_name if profile_context else None)

        rules: list[DomainEntityRule] = []
        if profile_context:
            rules.extend(profile_context.entity_rules)
        rules.extend(rule.to_rule() for rule in self.entity_rules)

        covariate_entity_types = list(self.covariate_entity_types or [])
        if not covariate_entity_types and profile_context:
            covariate_entity_types = list(profile_context.covariate_entity_types or [])

        resolved_prompt = self._resolve_prompt_value(
            root_dir,
            override=self.covariate_prompt,
            fallback=profile_context.covariate_prompt if profile_context else None,
            domain_name=domain_name,
        )
        resolved_description = self.covariate_description or (
            profile_context.covariate_description if profile_context else None
        )

        resolved_graph_prompt = self._resolve_prompt_value(
            root_dir,
            override=self.community_graph_prompt,
            fallback=profile_context.community_graph_prompt if profile_context else None,
            domain_name=domain_name,
        )
        resolved_text_prompt = self._resolve_prompt_value(
            root_dir,
            override=self.community_text_prompt,
            fallback=profile_context.community_text_prompt if profile_context else None,
            domain_name=domain_name,
        )

        covariate_type = self.covariate_type
        if covariate_type is None and profile_context:
            covariate_type = profile_context.covariate_type

        return DomainContext(
            domain_name=domain_name,
            entity_rules=tuple(rules),
            entity_tag_column=self.entity_tag_column,
            entity_primary_column=self.entity_primary_column,
            entity_profile_column=self.entity_profile_column,
            covariate_type=covariate_type,
            covariate_entity_types=tuple(covariate_entity_types) if covariate_entity_types else None,
            covariate_description=resolved_description,
            covariate_prompt=resolved_prompt,
            covariate_subject_tags_column=self.covariate_subject_tags_column,
            covariate_primary_tag_column=self.covariate_primary_tag_column,
            covariate_profile_column=self.covariate_profile_column,
            community_graph_prompt=resolved_graph_prompt,
            community_text_prompt=resolved_text_prompt,
        )

    @staticmethod
    def _resolve_prompt_value(
        root_dir: str,
        *,
        override: str | None,
        fallback: str | None,
        domain_name: str | None,
    ) -> str | None:
        """Resolve a prompt value, loading from disk when the path exists."""

        candidate = override or fallback
        if candidate is None:
            return None

        path = Path(candidate)
        if not path.is_absolute():
            path = Path(root_dir) / candidate
        if path.exists():
            candidate = path.read_text(encoding="utf-8")

        if domain_name:
            try:
                return candidate.format(domain_name=domain_name)
            except KeyError:
                return candidate
        return candidate
