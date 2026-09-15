"""业务逻辑层.

**这一层不依赖 FastAPI** —— 所有函数接收普通 Python 对象(或 SQLAlchemy 会话),
不接收 ``Request`` / ``UploadFile``. 好处是同一份逻辑可以被 HTTP 接口、
异步 Worker、CLI 脚本、消息队列消费者共同复用.

模块划分::

    parser/         PDF → 带坐标页码的文本块 → 清洗后的段落
    chunking/       段落 → 父子块
    embedding/      文本 → 向量(本地 BGE / 云端 API, 可切换)
    vectorstore/    向量 → 存储与检索(Chroma, 可替换)
    ingest.py       串联上述四步的入库编排
    document_service.py  文档的上传/查询/删除(业务规则)
"""

from __future__ import annotations

__all__: list[str] = []
