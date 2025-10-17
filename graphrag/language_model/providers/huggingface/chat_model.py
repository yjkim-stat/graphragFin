"""Chat model implementation for Hugging Face Inference endpoints."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from graphrag.language_model.response.base import (
    BaseModelOutput,
    BaseModelResponse,
)

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from graphrag.cache.pipeline_cache import PipelineCache
    from graphrag.config.models.language_model_config import LanguageModelConfig
    from huggingface_hub import AsyncInferenceClient, InferenceClient
    from huggingface_hub.utils import InferenceTimeoutError


class HuggingFaceModelOutput(BaseModelOutput):
    """Model output wrapper for Hugging Face responses."""


class HuggingFaceModelResponse(BaseModelResponse[BaseModel]):
    """Model response wrapper for Hugging Face responses."""


class HuggingFaceChatModel:
    """Hugging Face-based chat model."""

    def __init__(
        self,
        name: str,
        config: "LanguageModelConfig",
        cache: "PipelineCache | None" = None,
        *,
        task: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
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

        client_kwargs: dict[str, Any] = {"token": config.api_key, "timeout": timeout, "cache_dir":os.getenv('CACHE_DIR')}
        if base_url:
            client_kwargs["base_url"] = base_url.rstrip("/")

        self._client = InferenceClient(model=model, **client_kwargs)
        self._aclient = AsyncInferenceClient(model=model, **client_kwargs)
        self._timeout_error = InferenceTimeoutError

        self.task = task or getattr(config, "huggingface_task", None) or "text-generation"
        user_params = generation_kwargs or getattr(config, "huggingface_parameters", None) or {}
        self._generation_kwargs = self._resolve_generation_kwargs(user_params)

    def _resolve_generation_kwargs(self, user_params: dict[str, Any]) -> dict[str, Any]:
        resolved: dict[str, Any] = dict(user_params)
        max_tokens = (
            self.config.max_completion_tokens
            if self.config.max_completion_tokens is not None
            else self.config.max_tokens
        )
        if max_tokens is not None:
            resolved.setdefault("max_new_tokens", max_tokens)
        resolved.setdefault("temperature", self.config.temperature)
        resolved.setdefault("top_p", self.config.top_p)
        resolved.setdefault("return_full_text", False)
        if "do_sample" not in resolved:
            resolved["do_sample"] = (resolved.get("temperature", 0) or 0) > 0
        return resolved

    def _build_messages(self, prompt: str, history: list | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if history:
            for item in history:
                if isinstance(item, dict):
                    role = str(item.get("role", "user"))
                    content = str(item.get("content", ""))
                    messages.append({"role": role, "content": content})
                else:
                    messages.append({"role": "user", "content": str(item)})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_prompt(self, prompt: str, history: list | None) -> str:
        if not history:
            return prompt
        turns: list[str] = []
        for item in history:
            if isinstance(item, dict):
                role = str(item.get("role", "user")).upper()
                content = str(item.get("content", ""))
            else:
                role = "USER"
                content = str(item)
            turns.append(f"{role}: {content}")
        turns.append(f"USER: {prompt}")
        return "\n".join(turns)

    def _create_response(self, text: str, raw: Any, history: list) -> HuggingFaceModelResponse:
        output = HuggingFaceModelOutput(content=text, full_response=raw)
        return HuggingFaceModelResponse(output=output, history=history, cache_hit=False)

    def chat(self, prompt: str, history: list | None = None, **kwargs: Any) -> HuggingFaceModelResponse:
        messages = self._build_messages(prompt, history)
        try:
            if self.task == "chat-completion":
                response = self._client.chat_completion(
                    messages=messages,
                    stream=False,
                    **{**self._generation_kwargs, **kwargs},
                )
                choice = response.choices[0]
                text = getattr(choice.message, "content", None)
                if text is None:
                    text = choice.message.get("content") if isinstance(choice.message, dict) else ""
                history_out = [*messages, {"role": "assistant", "content": text}]
                return self._create_response(text or "", response.model_dump() if hasattr(response, "model_dump") else response, history_out)
            prompt_text = self._build_prompt(prompt, history)
            text = self._client.text_generation(
                prompt_text,
                stream=False,
                **{**self._generation_kwargs, **kwargs},
            )
            history_out = [*messages, {"role": "assistant", "content": text}]
            return self._create_response(text, text, history_out)
        except self._timeout_error as exc:  # pragma: no cover - passthrough
            raise TimeoutError(str(exc)) from exc

    async def achat(
        self,
        prompt: str,
        history: list | None = None,
        **kwargs: Any,
    ) -> HuggingFaceModelResponse:
        messages = self._build_messages(prompt, history)
        try:
            if self.task == "chat-completion":
                response = await self._aclient.chat_completion(
                    messages=messages,
                    stream=False,
                    **{**self._generation_kwargs, **kwargs},
                )
                choice = response.choices[0]
                text = getattr(choice.message, "content", None)
                if text is None:
                    text = choice.message.get("content") if isinstance(choice.message, dict) else ""
                history_out = [*messages, {"role": "assistant", "content": text}]
                raw = response.model_dump() if hasattr(response, "model_dump") else response
                return self._create_response(text or "", raw, history_out)
            prompt_text = self._build_prompt(prompt, history)
            text = await self._aclient.text_generation(
                prompt_text,
                stream=False,
                **{**self._generation_kwargs, **kwargs},
            )
            history_out = [*messages, {"role": "assistant", "content": text}]
            return self._create_response(text, text, history_out)
        except self._timeout_error as exc:  # pragma: no cover - passthrough
            raise TimeoutError(str(exc)) from exc

    def chat_stream(
        self,
        prompt: str,
        history: list | None = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        messages = self._build_messages(prompt, history)
        try:
            if self.task == "chat-completion":
                stream = self._client.chat_completion(
                    messages=messages,
                    stream=True,
                    **{**self._generation_kwargs, **kwargs},
                )
                for chunk in stream:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        text = getattr(delta, "content", None)
                        if text:
                            yield text
                return
            prompt_text = self._build_prompt(prompt, history)
            stream = self._client.text_generation(
                prompt_text,
                stream=True,
                **{**self._generation_kwargs, **kwargs},
            )
            for event in stream:
                token = getattr(event, "token", None)
                if token is not None:
                    text = getattr(token, "text", None)
                    if text and not getattr(token, "special", False):
                        yield text
        except self._timeout_error as exc:  # pragma: no cover - passthrough
            raise TimeoutError(str(exc)) from exc

    async def achat_stream(
        self,
        prompt: str,
        history: list | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        messages = self._build_messages(prompt, history)
        try:
            if self.task == "chat-completion":
                stream = self._aclient.chat_completion(
                    messages=messages,
                    stream=True,
                    **{**self._generation_kwargs, **kwargs},
                )
                async for chunk in stream:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        text = getattr(delta, "content", None)
                        if text:
                            yield text
                return
            prompt_text = self._build_prompt(prompt, history)
            stream = self._aclient.text_generation(
                prompt_text,
                stream=True,
                **{**self._generation_kwargs, **kwargs},
            )
            async for event in stream:
                token = getattr(event, "token", None)
                if token is not None:
                    text = getattr(token, "text", None)
                    if text and not getattr(token, "special", False):
                        yield text
        except self._timeout_error as exc:  # pragma: no cover - passthrough
            raise TimeoutError(str(exc)) from exc
