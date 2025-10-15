# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""LiteLLM completion/embedding logging wrapper."""

import logging
import time
from typing import Any

from graphrag.language_model.providers.litellm.types import (
    AsyncLitellmRequestFunc,
    LitellmRequestFunc,
)

logger = logging.getLogger(__name__)


def _build_logging_context(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Extract sanitized request context from liteLLM kwargs."""

    context: dict[str, Any] = {}

    raw_model = kwargs.get("model")
    if isinstance(raw_model, str):
        provider, _, model_name = raw_model.partition("/")
        if model_name:
            context["model_provider"] = provider
            context["model"] = model_name
        else:
            context["model"] = raw_model

    metadata = kwargs.get("metadata")
    if isinstance(metadata, dict):
        request_id = metadata.get("request_id")
        if isinstance(request_id, str) and request_id:
            context["request_id"] = request_id

    request_id = kwargs.get("request_id")
    if isinstance(request_id, str) and request_id:
        context.setdefault("request_id", request_id)

    return context


def _log_with_optional_context(level: str, message: str, context: dict[str, Any]) -> None:
    """Log at the requested level, attaching context when available."""

    if context:
        getattr(logger, level)(message, extra={"llm": context})
    else:
        getattr(logger, level)(message)


def with_logging(
    *,
    sync_fn: LitellmRequestFunc,
    async_fn: AsyncLitellmRequestFunc,
) -> tuple[LitellmRequestFunc, AsyncLitellmRequestFunc]:
    """
    Wrap the synchronous and asynchronous request functions with retries.

    Args
    ----
        sync_fn: The synchronous chat/embedding request function to wrap.
        async_fn: The asynchronous chat/embedding request function to wrap.
        model_config: The configuration for the language model.

    Returns
    -------
        A tuple containing the wrapped synchronous and asynchronous chat/embedding request functions.
    """

    def _wrapped_with_logging(**kwargs: Any) -> Any:
        context = _build_logging_context(kwargs)
        start_time = time.perf_counter()
        _log_with_optional_context("info", "llm_request_start", context)
        try:
            result = sync_fn(**kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000
            success_context = {**context, "duration_ms": duration_ms} if context else {
                "duration_ms": duration_ms
            }
            _log_with_optional_context("info", "llm_request_success", success_context)
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            failure_context = {**context, "duration_ms": duration_ms} if context else {
                "duration_ms": duration_ms
            }
            _log_with_optional_context(
                "exception",
                f"with_logging: Request failed with exception={e}",  # noqa: G004, TRY401
                failure_context,
            )
            raise

    async def _wrapped_with_logging_async(
        **kwargs: Any,
    ) -> Any:
        context = _build_logging_context(kwargs)
        start_time = time.perf_counter()
        _log_with_optional_context("info", "llm_request_start", context)
        try:
            result = await async_fn(**kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000
            success_context = {**context, "duration_ms": duration_ms} if context else {
                "duration_ms": duration_ms
            }
            _log_with_optional_context("info", "llm_request_success", success_context)
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            failure_context = {**context, "duration_ms": duration_ms} if context else {
                "duration_ms": duration_ms
            }
            _log_with_optional_context(
                "exception",
                f"with_logging: Async request failed with exception={e}",  # noqa: G004, TRY401
                failure_context,
            )
            raise

    return (_wrapped_with_logging, _wrapped_with_logging_async)
