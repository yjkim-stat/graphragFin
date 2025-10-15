"""Built-in domain intelligence profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .context import DomainContext
from .rules import DomainEntityRule
from graphrag.prompts.domain.finance import (
    FINANCE_COVARIATE_DESCRIPTION,
    FINANCE_COVARIATE_PROMPT,
    FINANCE_COMMUNITY_GRAPH_PROMPT,
    FINANCE_COMMUNITY_TEXT_PROMPT,
)


@dataclass(slots=True)
class DomainProfile:
    """Static bundle of domain intelligence assets."""

    domain_name: str
    entity_rules: tuple[DomainEntityRule, ...] = field(default_factory=tuple)
    covariate_type: str | None = None
    covariate_entity_types: tuple[str, ...] | None = None
    covariate_description: str | None = None
    covariate_prompt: str | None = None
    community_graph_prompt: str | None = None
    community_text_prompt: str | None = None

    def to_context(self) -> DomainContext:
        """Convert profile into a partial domain context."""

        return DomainContext(
            domain_name=self.domain_name,
            entity_rules=self.entity_rules,
            covariate_type=self.covariate_type,
            covariate_entity_types=self.covariate_entity_types,
            covariate_description=self.covariate_description,
            covariate_prompt=self.covariate_prompt,
            community_graph_prompt=self.community_graph_prompt,
            community_text_prompt=self.community_text_prompt,
        )


_FINANCE_RULES: tuple[DomainEntityRule, ...] = (
    DomainEntityRule(
        tag="commodity",
        match_types={"commodity", "material", "resource"},
        keywords={
            "copper",
            "aluminum",
            "nickel",
            "lithium",
            "oil",
            "gas",
            "crude",
            "corn",
            "wheat",
        },
        priority=10,
        primary=True,
    ),
    DomainEntityRule(
        tag="policy",
        match_types={"policy", "regulation", "tariff", "trade"},
        keywords={"tariff", "sanction", "quota", "export ban", "import ban"},
        priority=20,
    ),
    DomainEntityRule(
        tag="macro_indicator",
        match_types={"macro", "economic indicator"},
        keywords={
            "inflation",
            "interest rate",
            "gdp",
            "employment",
            "cpi",
            "pmi",
        },
        priority=30,
    ),
    DomainEntityRule(
        tag="supply_chain",
        keywords={"supply chain", "logistics", "port", "shipping", "freight"},
        priority=40,
    ),
    DomainEntityRule(
        tag="energy_infrastructure",
        keywords={"refinery", "pipeline", "grid", "power plant"},
        priority=50,
    ),
)

_PROFILES: Mapping[str, DomainProfile] = {
    "finance": DomainProfile(
        domain_name="Commodities and Futures",
        entity_rules=_FINANCE_RULES,
        covariate_type="finance_factor",
        covariate_entity_types=(
            "commodity",
            "policy",
            "macro",
            "market",
            "company",
            "infrastructure",
        ),
        covariate_description=FINANCE_COVARIATE_DESCRIPTION,
        covariate_prompt=FINANCE_COVARIATE_PROMPT,
        community_graph_prompt=FINANCE_COMMUNITY_GRAPH_PROMPT,
        community_text_prompt=FINANCE_COMMUNITY_TEXT_PROMPT,
    )
}


def get_profile(name: str | None) -> DomainProfile | None:
    """Return the profile associated with *name* if it exists."""

    if name is None:
        return None
    return _PROFILES.get(name.lower())


def get_profile_names() -> tuple[str, ...]:
    """Return the available profile names."""

    return tuple(sorted(_PROFILES.keys()))
