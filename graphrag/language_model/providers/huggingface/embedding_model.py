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


import numpy as np
import logging

logger = logging.getLogger(__name__)

class HuggingFaceEmbeddingModel:
    """Hugging Face-based embedding model."""

    def __init__(
        self,
        name: str,
        config: "LanguageModelConfig",
        cache: "PipelineCache | None" = None,
        *,
        feature_kwargs: dict[str, Any] | None = None,
        **kwargs,
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
        logger.info(f'data type: {type(data)}')

        if data is None:
            return []

        # torch.Tensor -> numpy
        if isinstance(data, torch.Tensor):
            arr = data.detach().cpu().float().numpy()
        else:
            # list/tuple/np.ndarray 전부 허용
            arr = np.array(data, dtype=float, copy=False)

        # 3D: (batch, tokens, dim)  -> 토큰 평균
        if arr.ndim == 3:
            # 일반적으로 feature-extraction이 여기 해당
            arr = arr.mean(axis=1)            # (batch, dim)
            arr = arr[0]                      # (dim,)

        # 2D: (batch, dim) 또는 (tokens, dim)
        elif arr.ndim == 2:
            b, d = arr.shape
            # 휴리스틱:
            # - 보통 dim은 64, 128, 384, 768 등 큰 값
            # - 배치가 1이면 확실히 (batch, dim)
            if b == 1:
                arr = arr[0]                  # (dim,)
            else:
                # b>1 이면 대부분 (batch, dim) 이므로 첫 배치 선택
                # 다만 (tokens, dim)일 가능성도 있으므로
                # tokens 평균을 원한다면 아래 한 줄로 교체:
                # arr = arr.mean(axis=0)
                arr = arr[0]

        # 1D: (dim,)
        elif arr.ndim == 1:
            pass

        else:
            raise ValueError(f"Unexpected embedding shape: {arr.shape}")

        return arr.tolist()
    def embed(self, text: str, **kwargs: Any) -> list[float]:
        logger.info(f'text:{text}')
        logger.info(f'kwargs:{kwargs}')
        embedding = self._pipeline(
            text,
            **{**self._feature_kwargs, **kwargs},
            truncation=True # 
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
