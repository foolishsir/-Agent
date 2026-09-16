"""LLM 提供方抽象.

为什么要抽象: 模型迭代极快, 今天 DeepSeek、明天通义、后天自部署 vLLM.
只要它们遵循 OpenAI 兼容协议, 换厂商就只是改 ``base_url`` 和 ``model`` 两个配置项.
把厂商名写进业务代码是技术债.

流式与非流式分成两个方法, 而不是用参数控制:
两者的**错误处理方式完全不同** —— 非流式调用失败就是失败, 整体重试即可;
流式调用一旦开始吐字, 中途失败不能重试(会重复输出), 只能把错误事件推给前端.
把这种差异编码进接口, 调用方就不会写错.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ChatMessage:
    """一条对话消息."""

    role: str  # system | user | assistant
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatUsage:
    """token 用量. 用于成本统计与评测."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResult:
    """一次完整的(非流式)生成结果."""

    content: str
    usage: ChatUsage = field(default_factory=ChatUsage)
    model: str = ""
    finish_reason: str = ""


@runtime_checkable
class LLMProvider(Protocol):
    """大模型提供方."""

    @property
    def name(self) -> str: ...

    @property
    def configured(self) -> bool: ...

    async def achat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """一次性生成(内部会重试)."""
        ...

    def astream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """流式生成, 逐段产出文本增量.

        设计成**同步返回 AsyncIterator** 而不是 ``async def``:
        这样调用方可以 ``async for chunk in provider.astream(...)`` 直接用,
        不需要先 await 一次. 注意: 网络异常的捕获要放在迭代过程中,
        因为此时请求还没发出.
        """
        ...
