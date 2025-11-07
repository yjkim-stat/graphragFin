"""Chat model implementation for local Hugging Face transformers models."""
from __future__ import annotations

import asyncio
import os
from threading import Lock, Thread

from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING, Any, Callable, ClassVar

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

import logging
import re
import json

# def extract_json(response_text: str):
#     """
#     Extracts the first JSON object found in a model-generated response string.
    
#     Parameters:
#         response_text (str): The full text output from the model.
    
#     Returns:
#         dict or None: Parsed JSON object if found, otherwise None.
#     """
#     # 중첩된 JSON도 매칭 가능한 정규식
#     pattern = r'\{(?:[^{}]|(?R))*\}'
    
#     match = re.search(pattern, response_text, re.DOTALL)
#     if not match:
#         return None  # JSON이 없을 때
    
#     json_str = match.group(0)
    
#     try:
#         return json.loads(json_str)
#     except json.JSONDecodeError:
#         # JSON 형식이 약간 깨져있을 경우에는 수동 보정 가능 (원하면 추가해줄게)
#         return None


from pydantic import BaseModel
from typing import Any

    
def extract_json(response_text: str):
    """
    Extract the first valid JSON object from a string by tracking brace depth.
    Works even when extra non-JSON text appears before or after the JSON.
    """
    start = None
    depth = 0

    for i, ch in enumerate(response_text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                json_str = response_text[start:i+1]
                return json_str
                # try:
                #     return json.loads(json_str)  # Return parsed JSON
                # except json.JSONDecodeError:
                #     return None

    # return None  # No valid JSON found

import json
import inspect
from typing import Any, cast
from pydantic import BaseModel, RootModel

# dict 래퍼 (루트 모델)
class DictModel(RootModel[dict[str, Any]]):
    pass

def _safe_extract_json_dict(text: str) -> dict[str, Any]:
    """
    text 안에서 유효한 JSON을 파싱해 dict로 반환.
    - 전체가 JSON 문자열이 아니거나 리스트/스칼라면 감싸서 dict로 반환.
    - 실패 시 빈 dict.
    """
    # 흔한 경우: ```json ... ``` 코드블록 제거
    t = text.strip()
    if t.startswith("```"):
        # ```json ...``` 또는 ``` ...``` 케이스 단순 처리
        first = t.find("\n")
        last = t.rfind("```")
        if first != -1 and last != -1:
            t = t[first + 1:last].strip()

    try:
        data = json.loads(t)
        if isinstance(data, dict):
            return data
        # 리스트/스칼라인 경우 감싸기
        return {"data": data}
    except Exception:
        return {}



logger = logging.getLogger(__name__)

class HuggingFaceModelOutput(BaseModelOutput):
    """Model output wrapper for Hugging Face responses."""


class HuggingFaceModelResponse(BaseModelResponse[BaseModel]):
    """Model response wrapper for Hugging Face responses."""


class HuggingFaceChatModel:
    """Hugging Face-based chat model."""

    _usage_lock: ClassVar[Lock] = Lock()
    _usage_stats: ClassVar[dict[str, dict[str, int]]] = {}

    def __init__(
        self,
        name: str,
        config: "LanguageModelConfig",
        cache: "PipelineCache | None" = None,
        *,
        task: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs,
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
            # dtype=torch.float16, 
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

    @classmethod
    def reset_usage(cls) -> None:
        """Clear aggregated usage statistics for all Hugging Face chat models."""

        with cls._usage_lock:
            cls._usage_stats = {}

    @classmethod
    def get_usage(cls) -> dict[str, dict[str, int]]:
        """Return cumulative usage statistics per registered chat model."""

        with cls._usage_lock:
            return {name: stats.copy() for name, stats in cls._usage_stats.items()}

    @classmethod
    def get_total_usage(cls) -> dict[str, int]:
        """Return aggregate usage totals across all chat model instances."""

        with cls._usage_lock:
            totals = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "llm_calls": 0,
            }
            for stats in cls._usage_stats.values():
                totals["prompt_tokens"] += stats.get("prompt_tokens", 0)
                totals["completion_tokens"] += stats.get("completion_tokens", 0)
                totals["total_tokens"] += stats.get("total_tokens", 0)
                totals["llm_calls"] += stats.get("llm_calls", 0)
            return totals

    def _record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token usage for the current chat invocation."""

        total_tokens = prompt_tokens + completion_tokens
        with self.__class__._usage_lock:
            stats = self.__class__._usage_stats.setdefault(
                self.name,
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "llm_calls": 0,
                },
            )
            stats["prompt_tokens"] += prompt_tokens
            stats["completion_tokens"] += completion_tokens
            stats["total_tokens"] += total_tokens
            stats["llm_calls"] += 1

    def _build_usage_metrics(
        self, prompt_text: str, completion_text: str
    ) -> dict[str, int]:
        """Calculate token usage metrics for a generated completion."""

        prompt_tokens = len(self._tokenizer.encode(prompt_text))
        completion_tokens = (
            len(self._tokenizer.encode(completion_text)) if completion_text else 0
        )
        self._record_usage(prompt_tokens, completion_tokens)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "llm_calls": 1,
        }

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

    # def _create_response(
    #     self,
    #     text: str,
    #     raw: Any,
    #     history: list,
    #     metrics: dict[str, Any] | None = None,
    #     **kwargs
    # ) -> HuggingFaceModelResponse:
    #     full_response = raw if isinstance(raw, dict) or raw is None else {"text": str(raw)}
    #     output = HuggingFaceModelOutput(content=text, full_response=full_response)

    #     parsed_response = None

    #     if kwargs.get("json", False) is True:
    #         model_cls = kwargs.get("response_format")
    #         if inspect.isclass(model_cls) and issubclass(model_cls, BaseModel):
    #             # 문자열을 바로 모델로 검증 (권장, v2)
    #             try:
    #                 parsed_response = model_cls.model_validate_json(text)  # type: ignore[attr-defined]
    #             except Exception:
    #                 # 모델이 JSON이 아닌 프리앰블/설명과 함께 올 수 있으니 안전 파싱 후 검증
    #                 data = _safe_extract_json_dict(text)
    #                 parsed_response = model_cls.model_validate(data)
    #         else:
    #             # 스키마가 없으면 dict로 파싱해 넣기 (아래 방법 2와 동일)
    #             data = _safe_extract_json_dict(text)
    #             parsed_response = DictModel.model_validate({"__root__": data})

    #     return HuggingFaceModelResponse(
    #         output=output,
    #         history=history,
    #         cache_hit=False,
    #         metrics=metrics,
    #         parsed_response=parsed_response,
    #     )


    def _create_response(
        self,
        text: str,
        raw: Any,
        history: list,
        metrics: dict[str, Any] | None = None,
        **kwargs
    ) -> HuggingFaceModelResponse:
        # full_response 구성
        full_response = raw if isinstance(raw, dict) or raw is None else {"text": str(raw)}
        output = HuggingFaceModelOutput(content=text, full_response=full_response)

        parsed_response: BaseModel | None = None

        # OpenAI 호환 플래그들: json, name, json_model
        want_json: bool = bool(kwargs.get("json", False))
        # 모델 클래스는 json_model 우선, 없으면 response_format 호환
        model_cls = kwargs.get("json_model") or kwargs.get("response_format")

        logger.info(f'want_json:{want_json}')
        logger.info(f'model_cls:{model_cls}')
        logger.info(f'text:{text}\n{_safe_extract_json_dict(text)}')
        if want_json:
            if inspect.isclass(model_cls) and issubclass(model_cls, BaseModel):
                # 스키마가 있으면: 문자열을 바로 모델로 검증 (v2)
                try:
                    parsed_response = model_cls.model_validate_json(text)  # type: ignore[attr-defined]
                except Exception:
                    # 문자열에 프리앰블/코드블록 등이 섞였을 수 있으니 안전 파싱 후 검증
                    data = _safe_extract_json_dict(text)
                    parsed_response = model_cls.model_validate(data)
            else:
                # 스키마가 없으면: dict로 파싱 후 RootModel 래퍼에 담아 BaseModel 보장
                data = _safe_extract_json_dict(text)
                parsed_response = DictModel.model_validate(data)

        return HuggingFaceModelResponse(
            output=output,
            history=history,
            cache_hit=False,
            metrics=metrics,
            parsed_response=parsed_response,
        )


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
            for _k in ['name', 'json', 'json_model']:
                if gen_kwargs.pop(_k, False):
                    # NOTE ('name') HF generation에 필요없고 OpenAI에서만 지원하는 구조임.
                    logger.info(f'{_k} removed from gen_kwargs')
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
        kwargs_reports = dict()
        kwargs_reports['json'] = kwargs.pop('json', None)
        kwargs_reports['name'] = kwargs.pop('name', None)
        kwargs_reports['json_model'] = kwargs.pop('json_model', None)
        messages = self._build_messages(prompt, history)
        prompt_text = self._prepare_prompt(prompt, history, messages)
        if kwargs_reports['json']:
            prompt_text += """\n\nAnswer in JSON only.\n\nIf your output contains anything outside of the JSON object, your response is invalid."""

        text = self._generate_text(prompt_text, **kwargs)
        history_out = [*messages, {"role": "assistant", "content": text}]
        metrics = self._build_usage_metrics(prompt_text, text)

        kwargs_response = dict(**kwargs_reports)
        return self._create_response(text, text, history_out, metrics, **kwargs_response)

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

        logger.info(f'messages\n\t{messages}')
        logger.info(f'prompt_text\n\t{prompt_text}')
        for _k in ['model_parameters']:
            if kwargs.pop(_k, False):
                # NOTE ('name') HF generation에 필요없고 OpenAI에서만 지원하는 구조임.
                logger.info(f'{_k} removed from kwargs')

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
