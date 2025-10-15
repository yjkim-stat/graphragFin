# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

from pathlib import Path

from pydantic import BaseModel, Field

from graphrag.config.defaults import graphrag_config_defaults
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.domain.context import DomainContext


class ClaimExtractionConfig(BaseModel):
    """Configuration section for claim extraction."""

    enabled: bool = Field(
        description="Whether claim extraction is enabled.",
        default=graphrag_config_defaults.extract_claims.enabled,
    )
    model_id: str = Field(
        description="The model ID to use for claim extraction.",
        default=graphrag_config_defaults.extract_claims.model_id,
    )
    prompt: str | None = Field(
        description="The claim extraction prompt to use.",
        default=graphrag_config_defaults.extract_claims.prompt,
    )
    description: str = Field(
        description="The claim description to use.",
        default=graphrag_config_defaults.extract_claims.description,
    )
    max_gleanings: int = Field(
        description="The maximum number of entity gleanings to use.",
        default=graphrag_config_defaults.extract_claims.max_gleanings,
    )
    strategy: dict | None = Field(
        description="The override strategy to use.",
        default=graphrag_config_defaults.extract_claims.strategy,
    )

    def resolved_strategy(
        self,
        root_dir: str,
        model_config: LanguageModelConfig,
        domain_context: DomainContext | None = None,
    ) -> dict:
        """Get the resolved claim extraction strategy."""
        if self.strategy:
            return self.strategy

        prompt_override = None
        if domain_context and domain_context.covariate_prompt:
            prompt_override = domain_context.covariate_prompt
        elif self.prompt:
            prompt_override = (Path(root_dir) / self.prompt).read_text(
                encoding="utf-8"
            )

        description = self.description
        if domain_context and domain_context.covariate_description:
            description = domain_context.covariate_description

        return {
            "llm": model_config.model_dump(),
            "extraction_prompt": prompt_override,
            "claim_description": description,
            "max_gleanings": self.max_gleanings,
        }
