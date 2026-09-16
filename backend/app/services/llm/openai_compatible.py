"""OpenAI 兼容协议的 LLM 实现.

DeepSeek / 通义(兼容模式) / 硅基流动 / vLLM / Ollama 都遵循这套协议,
所以一个实现就能覆盖绝大多数场景.

重试策略
--------
只对**幂等且可恢复**的错误重试: 超时、连接错误、429 限流、5xx.
**不重试** 4xx(除 429)—— 400/401/404 是配置问题, 重试一百次也一样失败,
只会白白拖慢响应并浪费用户等待时间. 这是很常见的过度重试反模式.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.core.config import settings
from app.core.exceptions import LLMError, LLMNotConfiguredError, LLMTimeoutError
from app.core.logging import get_logger, log_kv
from app.services.llm.base import ChatMessage, ChatResult, ChatUsage

logger = get_logger("docmind.llm")

#: 这些 HTTP 状态码值得重试: 限流与服务端临时故障
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatibleLLM:
    """基于 openai SDK 的通用实现."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 2,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client: Any | None = None

    # ------------------------------------------------------------------ #
    # 基本信息
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return f"{self._base_url}::{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @property
    def configured(self) -> bool:
        return bool(self._api_key.strip())

    def _ensure_client(self) -> Any:
        if self._client is None:
            if not self.configured:
                raise LLMNotConfiguredError()
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover
                raise LLMError("未安装 openai 包, 请执行: pip install openai") from exc

            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
                # 关闭 SDK 自带重试, 由我们自己的策略统一管理 ——
                # 两套重试叠加会导致等待时间指数级放大(用户感觉"卡死了")
                max_retries=0,
            )
        return self._client

    # ------------------------------------------------------------------ #
    # 非流式
    # ------------------------------------------------------------------ #
    async def achat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        client = self._ensure_client()
        payload = {
            "model": self._model,
            "messages": [m.to_dict() for m in messages],
            "temperature": self._temperature if temperature is None else temperature,
            "max_tokens": self._max_tokens if max_tokens is None else max_tokens,
        }

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await client.chat.completions.create(**payload)
                choice = response.choices[0]
                usage = getattr(response, "usage", None)
                return ChatResult(
                    content=choice.message.content or "",
                    usage=ChatUsage(
                        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                        total_tokens=getattr(usage, "total_tokens", 0) or 0,
                    ),
                    model=getattr(response, "model", self._model),
                    finish_reason=getattr(choice, "finish_reason", "") or "",
                )
            except Exception as exc:  # noqa: BLE001 - SDK 异常层次不稳定
                last_error = exc
                if not self._should_retry(exc) or attempt >= self._max_retries:
                    break
                log_kv(
                    logger,
                    "llm.retry",
                    attempt=attempt + 1,
                    max=self._max_retries,
                    error=type(exc).__name__,
                )

        raise self._translate_error(last_error)

    # ------------------------------------------------------------------ #
    # 流式
    # ------------------------------------------------------------------ #
    async def astream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """流式生成.

        **不做重试**: 一旦开始向用户吐字, 重试会造成重复输出.
        中途失败只能把错误抛给调用方, 由调用方向前端推一个 error 事件.
        """
        client = self._ensure_client()
        try:
            stream = await client.chat.completions.create(
                model=self._model,
                messages=[m.to_dict() for m in messages],
                temperature=self._temperature if temperature is None else temperature,
                max_tokens=self._max_tokens if max_tokens is None else max_tokens,
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise self._translate_error(exc) from exc

        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # 有些兼容实现会返回空 delta(例如只带 role 的首包), 要跳过
                content = getattr(delta, "content", None)
                if content:
                    yield content
        except Exception as exc:  # noqa: BLE001
            raise self._translate_error(exc) from exc

    # ------------------------------------------------------------------ #
    # 错误处理
    # ------------------------------------------------------------------ #
    @staticmethod
    def _should_retry(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status is not None:
            return int(status) in _RETRYABLE_STATUS
        # 无状态码的通常是网络层问题(超时/连接中断), 值得重试
        name = type(exc).__name__.lower()
        return any(k in name for k in ("timeout", "connection", "connect"))

    @staticmethod
    def _translate_error(exc: Exception | None) -> LLMError:
        """把 SDK 的异常翻译成我们自己的业务异常.

        为什么要翻译: 直接把 openai 的异常抛到接口层, 会让 API 响应里
        暴露第三方库的类名和内部结构; 而且前端无法据此做差异化处理.
        """
        if exc is None:
            return LLMError("大模型调用失败: 未知原因")

        name = type(exc).__name__.lower()
        if "timeout" in name:
            return LLMTimeoutError(f"大模型调用超时(>{settings.llm_timeout}s)")

        status = getattr(exc, "status_code", None)
        if status == 401:
            return LLMError("API Key 无效或已过期(401), 请在「设置」里检查")
        if status == 402:
            return LLMError("账户余额不足(402), 请先充值")
        if status == 404:
            return LLMError("模型不存在(404), 请检查模型名称是否正确")
        if status == 429:
            return LLMError("触发限流(429), 请稍后重试或降低并发")
        if status is not None:
            return LLMError(f"大模型返回错误(HTTP {status}): {exc}")

        return LLMError(f"大模型调用失败: {exc}")
