"""Domain entity rule definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DomainEntityRule:
    """Simple matching rule for entity domain tagging."""

    tag: str
    match_types: set[str] = field(default_factory=set)
    keywords: set[str] = field(default_factory=set)
    match_titles: set[str] = field(default_factory=set)
    attribute_overrides: dict[str, Any] = field(default_factory=dict)
    priority: int = 100
    primary: bool = False

    def matches(self, *, title: str, entity_type: str | None, description: str | None) -> bool:
        """Return True when the rule applies to the given entity fields."""

        normalized_type = (entity_type or "").strip().lower()
        normalized_title = (title or "").strip().lower()
        normalized_description = (description or "").strip().lower()

        if self.match_types and normalized_type in self.match_types:
            return True

        haystack = " ".join(part for part in [normalized_title, normalized_description] if part)
        if self.keywords:
            for keyword in self.keywords:
                if keyword in haystack:
                    return True

        if self.match_titles and any(match in normalized_title for match in self.match_titles):
            return True

        return False

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> "DomainEntityRule":
        """Create a rule from configuration data."""

        return cls(
            tag=data["tag"],
            match_types={value.strip().lower() for value in data.get("match_types", [])},
            keywords={value.strip().lower() for value in data.get("keywords", [])},
            match_titles={value.strip().lower() for value in data.get("match_titles", [])},
            attribute_overrides=data.get("attribute_overrides", {}) or {},
            priority=data.get("priority", 100),
            primary=bool(data.get("primary", False)),
        )
