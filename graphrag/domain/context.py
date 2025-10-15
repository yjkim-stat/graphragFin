"""Runtime types for domain-intelligence enrichment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .rules import DomainEntityRule


@dataclass(slots=True)
class DomainContext:
    """Resolved domain-intelligence configuration."""

    domain_name: str | None
    entity_rules: Sequence[DomainEntityRule] = field(default_factory=tuple)
    entity_tag_column: str = "domain_tags"
    entity_primary_column: str | None = "domain_primary_tag"
    entity_profile_column: str | None = "domain_profile"
    covariate_type: str | None = None
    covariate_entity_types: Sequence[str] | None = None
    covariate_description: str | None = None
    covariate_prompt: str | None = None
    covariate_subject_tags_column: str = "subject_domain_tags"
    covariate_primary_tag_column: str | None = "subject_domain_primary_tag"
    covariate_profile_column: str | None = "domain_profile"
    community_graph_prompt: str | None = None
    community_text_prompt: str | None = None

    def has_entity_rules(self) -> bool:
        """Return True when there are entity rules to evaluate."""

        return bool(self.entity_rules)

    def iter_rules(self) -> Iterable[DomainEntityRule]:
        """Iterate through rules preserving definition order."""

        return iter(self.entity_rules)

    def wants_covariate_prompt_override(self) -> bool:
        """Return True when a domain specific covariate prompt is available."""

        return self.covariate_prompt is not None or self.covariate_description is not None

    def wants_covariate_type_override(self) -> bool:
        """Return True when covariate type should be overwritten."""

        return self.covariate_type is not None

    def wants_covariate_entity_types(self) -> bool:
        """Return True when covariate extraction should restrict entity types."""

        return self.covariate_entity_types is not None and len(self.covariate_entity_types) > 0

    def wants_community_prompt_override(self) -> bool:
        """Return True when community prompts should be replaced."""

        return (self.community_graph_prompt is not None) or (
            self.community_text_prompt is not None
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialize minimal context metadata."""

        return {
            "domain_name": self.domain_name,
            "entity_tag_column": self.entity_tag_column,
            "entity_primary_column": self.entity_primary_column,
            "entity_profile_column": self.entity_profile_column,
            "covariate_type": self.covariate_type,
            "covariate_entity_types": list(self.covariate_entity_types or []),
        }
