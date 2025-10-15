"""Embedding model implementation for Hugging Face Inference endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from graphrag.cache.pipeline_cache import PipelineCache
    from graphrag.config.models.language_model_config import LanguageModelConfig
    from huggingface_hub import AsyncInferenceClient, InferenceClient
    from huggingface_hub.utils import InferenceTimeoutError


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
            from huggingface_hub import AsyncInferenceClient, InferenceClient
            from huggingface_hub.utils import InferenceTimeoutError
        except ImportError as exc:  # pragma: no cover - import guard
            msg = (
                "huggingface-hub is required to use the Hugging Face model providers. "
                "Install it with `pip install huggingface-hub`."
            )
            raise ImportError(msg) from exc

        self.name = name
        self.config = config
        self.cache = cache.child(self.name) if cache else None

        model = config.deployment_name or config.model
        timeout = config.request_timeout or None
        base_url = config.api_base or None

        client_kwargs: dict[str, Any] = {"token": config.api_key, "timeout": timeout}
        if base_url:
            client_kwargs["base_url"] = base_url.rstrip("/")

        self._client = InferenceClient(model=model, **client_kwargs)
        self._aclient = AsyncInferenceClient(model=model, **client_kwargs)
        self._timeout_error = InferenceTimeoutError

        self._feature_kwargs = feature_kwargs or getattr(config, "huggingface_parameters", None) or {}

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
        try:
            embedding = self._client.feature_extraction(
                text,
                **{**self._feature_kwargs, **kwargs},
            )
            return self._normalize_embedding(embedding)
        except self._timeout_error as exc:  # pragma: no cover - passthrough
            raise TimeoutError(str(exc)) from exc

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        try:
            embedding = await self._aclient.feature_extraction(
                text,
                **{**self._feature_kwargs, **kwargs},
            )
            return self._normalize_embedding(embedding)
        except self._timeout_error as exc:  # pragma: no cover - passthrough
            raise TimeoutError(str(exc)) from exc

    def embed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        return [self.embed(text, **kwargs) for text in text_list]

    async def aembed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        results: list[list[float]] = []
        for text in text_list:
            results.append(await self.aembed(text, **kwargs))
        return results
