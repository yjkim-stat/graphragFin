"""Embedding model implementation for local Hugging Face transformers models."""
from __future__ import annotations

import asyncio
import os

from typing import TYPE_CHECKING, Any

try:
    import torch
except ImportError as exc:  # pragma: no cover - import guard
    msg = "PyTorch is required to run local Hugging Face models. Install it with `pip install torch`."
    raise ImportError(msg) from exc

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from graphrag.cache.pipeline_cache import PipelineCache
    from graphrag.config.models.language_model_config import LanguageModelConfig


class HuggingFaceEmbeddingModel:
    """Hugging Face-based embedding model."""

    def __init__(
        self,
        name: str,
        config: "LanguageModelConfig",
        cache: "PipelineCache | None" = None,
        *,
        feature_kwargs: dict[str, Any] | None = None,
    ) -> None:
        try:
            from transformers import pipeline  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # pragma: no cover - import guard
            msg = (
                "transformers is required to use the Hugging Face model providers. "
                "Install it with `pip install transformers`."
            )
            raise ImportError(msg) from exc

        self.name = name
        self.config = config
        self.cache = cache.child(self.name) if cache else None

        model = config.deployment_name or config.model
        cache_dir = os.getenv("CACHE_DIR") or None

        raw_params = dict(
            feature_kwargs or getattr(config, "huggingface_parameters", None) or {}
        )
        pipeline_kwargs = raw_params.pop("pipeline_kwargs", {})
        tokenizer_kwargs = raw_params.pop("tokenizer_kwargs", {})
        model_kwargs = raw_params.pop("model_kwargs", {})

        if cache_dir:
            tokenizer_kwargs.setdefault("cache_dir", cache_dir)
            model_kwargs.setdefault("cache_dir", cache_dir)

        pipeline_kwargs = dict(pipeline_kwargs)
        if "device" not in pipeline_kwargs:
            pipeline_kwargs["device"] = 0 if torch.cuda.is_available() else -1

        self._pipeline = pipeline(
            "feature-extraction",
            model=model,
            tokenizer=model,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            **pipeline_kwargs,
        )

        self._feature_kwargs = raw_params

    def _normalize_embedding(self, data: Any) -> list[float]:
        if data is None:
            return []
        if isinstance(data, (list, tuple)) and data and isinstance(data[0], (list, tuple)):
            # take first element for sequence outputs (e.g., sentence transformers)
            vector = data[0]
        else:
            vector = data
        if isinstance(vector, (list, tuple)):
            return [float(x) for x in vector]
        return [float(vector)]

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        embedding = self._pipeline(
            text,
            **{**self._feature_kwargs, **kwargs},
        )
        return self._normalize_embedding(embedding)

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        return await asyncio.to_thread(self.embed, text, **kwargs)

    def embed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        return [self.embed(text, **kwargs) for text in text_list]

    async def aembed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        results: list[list[float]] = []
        for text in text_list:
            results.append(await self.aembed(text, **kwargs))
        return results
