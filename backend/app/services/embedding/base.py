"""Embedding Provider 抽象.

为什么要有这层抽象
------------------
模型迭代极快, 今天用本地 BGE, 明天可能换云端 API, 后天可能换自部署的 vLLM.
如果业务代码里直接 ``from sentence_transformers import SentenceTransformer``,
换模型就是一次全项目重构. 抽象成协议后, 换模型只改一个工厂函数 + 一行配置.

**query 与 passage 必须分开编码**
--------------------------------
这是文本检索里最容易被忽略、代价却最大的细节.

双塔检索模型对"查询"和"文档"的编码**不必然对称**. 以 BGE 中文系列为例,
训练时 query 侧会加一句指令前缀(如 "为这个句子生成表示以用于检索相关文章："),
passage 侧不加.

如果代码里对两者用同一套逻辑编码 —— 很多人的写法就是这样 —— 检索效果会**静默劣化**:
不报错、不崩溃, 只是召回率悄悄变低. 这种 bug 极难发现, 因为一切"看起来正常".

因此协议里刻意把方法拆成 ``encode_query`` 和 ``encode_passages`` 两个,
从接口层面强制调用方区分两者.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """文本向量化提供方."""

    @property
    def name(self) -> str:
        """用于日志与调试的标识, 如 ``local:BAAI/bge-small-zh-v1.5``."""
        ...

    @property
    def dim(self) -> int:
        """向量维度. 建向量库集合时必须与之一致."""
        ...

    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """编码文档片段(passage 侧).

        用于入库阶段. 必须支持批量以摊薄模型调用开销.
        """
        ...

    def encode_query(self, text: str) -> list[float]:
        """编码用户查询(query 侧).

        与 ``encode_passages`` **可能使用不同的预处理**(见模块文档).
        """
        ...
