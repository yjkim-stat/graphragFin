"""Chat model implementation for local Hugging Face transformers models."""
from __future__ import annotations

import asyncio
import os
from threading import Thread

from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING, Any, Callable

try:
    import torch
except ImportError as exc:  # pragma: no cover - import guard
    msg = "PyTorch is required to run local Hugging Face models. Install it with `pip install torch`."
    raise ImportError(msg) from exc
from pydantic import BaseModel

from graphrag.language_model.response.base import (
    BaseModelOutput,
    BaseModelResponse,
)

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from graphrag.cache.pipeline_cache import PipelineCache
    from graphrag.config.models.language_model_config import LanguageModelConfig


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
            from transformers import (  # pylint: disable=import-outside-toplevel
                AutoModelForCausalLM,
                AutoTokenizer,
                TextIteratorStreamer,
                BitsAndBytesConfig
            )
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

        self.task = task or getattr(config, "huggingface_task", None) or "text-generation"
        raw_params = dict(
            generation_kwargs or getattr(config, "huggingface_parameters", None) or {}
        )
        tokenizer_kwargs = raw_params.pop("tokenizer_kwargs", {})
        model_kwargs = raw_params.pop("model_kwargs", {})
        self._streamer_factory: Callable[..., TextIteratorStreamer] = TextIteratorStreamer

        if cache_dir:
            tokenizer_kwargs.setdefault("cache_dir", cache_dir)
            model_kwargs.setdefault("cache_dir", cache_dir)

        self._tokenizer = AutoTokenizer.from_pretrained(model, **tokenizer_kwargs)
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        quantization_config = {
            'load_in_4bit': True,
            'load_in_8bit': False,
            'bnb_4bit_use_double_quant': True,
            'bnb_4bit_compute_dtype': 'float16',
            'bnb_4bit_quant_type': 'nf4',
        }
        self._model = AutoModelForCausalLM.from_pretrained(
            model, 
            dtype=torch.float16, 
            trust_remote_code=True, 
            quantization_config=BitsAndBytesConfig(**quantization_config), 
            # cache_dir=os.getenv('CACHE_DIR'),
            attn_implementation=os.getenv('ATTN_IMPLEMENTATION', "flash_attention_2"),
            token=os.getenv('HF_TOKEN'),
            **model_kwargs
            )
        self._model.eval()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

        self._generation_kwargs = self._resolve_generation_kwargs(raw_params)

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

    def _prepare_prompt(self, prompt: str, history: list | None, messages: list[dict[str, str]]) -> str:
        if self.task == "chat-completion" and hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return self._build_prompt(prompt, history)

    def _tokenize_prompt(self, prompt_text: str) -> dict[str, torch.Tensor]:
        encoded = self._tokenizer(prompt_text, return_tensors="pt")
        return {key: value.to(self._device) for key, value in encoded.items()}

    def _generate_text(self, prompt_text: str, **kwargs: Any) -> str:
        gen_kwargs = {**self._generation_kwargs, **kwargs}
        return_full_text = bool(gen_kwargs.pop("return_full_text", False))
        if self._tokenizer.pad_token_id is not None:
            gen_kwargs.setdefault("pad_token_id", self._tokenizer.pad_token_id)
        eos_token_id = getattr(self._model.config, "eos_token_id", None)
        if eos_token_id is not None:
            gen_kwargs.setdefault("eos_token_id", eos_token_id)
        inputs = self._tokenize_prompt(prompt_text)
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        prompt_length = inputs["input_ids"].shape[-1]
        generated = output_ids[:, prompt_length:]
        text = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        if return_full_text:
            return f"{prompt_text}{text}"
        return text

    def _stream_generate(self, prompt_text: str, **kwargs: Any) -> Generator[str, None, None]:
        gen_kwargs = {**self._generation_kwargs, **kwargs}
        return_full_text = bool(gen_kwargs.pop("return_full_text", False))
        if self._tokenizer.pad_token_id is not None:
            gen_kwargs.setdefault("pad_token_id", self._tokenizer.pad_token_id)
        eos_token_id = getattr(self._model.config, "eos_token_id", None)
        if eos_token_id is not None:
            gen_kwargs.setdefault("eos_token_id", eos_token_id)
        streamer = self._streamer_factory(
            self._tokenizer,
            skip_prompt=not return_full_text,
            skip_special_tokens=True,
        )
        inputs = self._tokenize_prompt(prompt_text)
        thread = Thread(
            target=self._model.generate,
            kwargs={**inputs, **gen_kwargs, "streamer": streamer},
            daemon=True,
        )
        thread.start()
        try:
            for chunk in streamer:
                yield chunk
        finally:
            thread.join()

    def chat(self, prompt: str, history: list | None = None, **kwargs: Any) -> HuggingFaceModelResponse:
        messages = self._build_messages(prompt, history)
        prompt_text = self._prepare_prompt(prompt, history, messages)
        text = self._generate_text(prompt_text, **kwargs)
        history_out = [*messages, {"role": "assistant", "content": text}]
        return self._create_response(text, text, history_out)

    async def achat(
        self,
        prompt: str,
        history: list | None = None,
        **kwargs: Any,
    ) -> HuggingFaceModelResponse:
        return await asyncio.to_thread(self.chat, prompt, history, **kwargs)

    def chat_stream(
        self,
        prompt: str,
        history: list | None = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        messages = self._build_messages(prompt, history)
        prompt_text = self._prepare_prompt(prompt, history, messages)
        yield from self._stream_generate(prompt_text, **kwargs)

    async def achat_stream(
        self,
        prompt: str,
        history: list | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()
        loop = asyncio.get_running_loop()

        def enqueue_stream() -> None:
            try:
                for chunk in self.chat_stream(prompt, history, **kwargs):
                    asyncio.run_coroutine_threadsafe(queue.put(str(chunk)), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop)

        Thread(target=enqueue_stream, daemon=True).start()

        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if item is not None:
                yield item
