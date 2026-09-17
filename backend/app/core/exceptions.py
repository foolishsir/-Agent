"""统一异常体系.

设计要点
--------
1. 业务异常与 HTTP 状态码解耦: 抛出的是 ``AppException``, 由全局异常处理器
   统一翻译成 HTTP 响应, 业务代码里不再散落 ``raise HTTPException``.
2. 每个异常都带机器可读的 ``code``, 前端据此做差异化提示(而不是解析中文文案).
3. 预留 ``detail`` 字段承载调试信息, 生产环境可决定是否透出.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    """机器可读的错误码.

    用 ``StrEnum`` 而非 ``class ErrorCode(str, Enum)``:
    前者是 Python 3.11+ 的标准写法, 成员本身就是 str, 可直接参与
    字符串比较与 JSON 序列化, 且 ``str(ErrorCode.OK)`` 得到 ``'OK'``
    而不是 ``'ErrorCode.OK'`` —— 后者在日志里非常难看.
    """

    OK = "OK"

    # 通用
    PARAM_INVALID = "PARAM_INVALID"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    RATE_LIMITED = "RATE_LIMITED"
    INTERNAL_ERROR = "INTERNAL_ERROR"

    # 文档
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    DUPLICATE_DOCUMENT = "DUPLICATE_DOCUMENT"
    DOC_PARSE_FAILED = "DOC_PARSE_FAILED"
    DOC_OCR_REQUIRED = "DOC_OCR_REQUIRED"
    DOC_NOT_READY = "DOC_NOT_READY"
    DOC_NO_TEXT = "DOC_NO_TEXT"

    # 检索 / 向量
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    VECTOR_STORE_ERROR = "VECTOR_STORE_ERROR"
    RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
    NO_CONTEXT_FOUND = "NO_CONTEXT_FOUND"

    # LLM
    LLM_NOT_CONFIGURED = "LLM_NOT_CONFIGURED"
    LLM_ERROR = "LLM_ERROR"
    LLM_TIMEOUT = "LLM_TIMEOUT"

    # 语音
    SPEECH_NOT_CONFIGURED = "SPEECH_NOT_CONFIGURED"
    SPEECH_ERROR = "SPEECH_ERROR"


class AppException(Exception):
    """所有业务异常的基类."""

    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    http_status: int = 500
    message: str = "服务内部错误"

    def __init__(
        self,
        message: str | None = None,
        *,
        code: ErrorCode | None = None,
        http_status: int | None = None,
        detail: Any = None,
    ) -> None:
        self.message = message or self.message
        if code is not None:
            self.code = code
        if http_status is not None:
            self.http_status = http_status
        self.detail = detail
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code.value, "message": self.message}
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"<{type(self).__name__} code={self.code.value} message={self.message!r}>"


# ---------------------------------------------------------------------- #
# 通用异常
# ---------------------------------------------------------------------- #
class ParamInvalidError(AppException):
    code = ErrorCode.PARAM_INVALID
    http_status = 400
    message = "请求参数不合法"


class NotFoundError(AppException):
    code = ErrorCode.NOT_FOUND
    http_status = 404
    message = "资源不存在"


class ConflictError(AppException):
    code = ErrorCode.CONFLICT
    http_status = 409
    message = "资源冲突"


# ---------------------------------------------------------------------- #
# 文档链路异常
# ---------------------------------------------------------------------- #
class FileTooLargeError(AppException):
    code = ErrorCode.FILE_TOO_LARGE
    http_status = 413
    message = "文件超过大小限制"


class UnsupportedFileTypeError(AppException):
    code = ErrorCode.UNSUPPORTED_FILE_TYPE
    http_status = 415
    message = "不支持的文件类型"


class DuplicateDocumentError(AppException):
    code = ErrorCode.DUPLICATE_DOCUMENT
    http_status = 409
    message = "该文档已存在"


class DocumentParseError(AppException):
    code = ErrorCode.DOC_PARSE_FAILED
    http_status = 422
    message = "文档解析失败"


class DocumentNotReadyError(AppException):
    code = ErrorCode.DOC_NOT_READY
    http_status = 409
    message = "文档尚未完成向量化, 暂时无法问答"


# ---------------------------------------------------------------------- #
# 检索 / 向量异常
# ---------------------------------------------------------------------- #
class EmbeddingError(AppException):
    code = ErrorCode.EMBEDDING_FAILED
    http_status = 500
    message = "文本向量化失败"


class VectorStoreError(AppException):
    code = ErrorCode.VECTOR_STORE_ERROR
    http_status = 500
    message = "向量库操作失败"


class RetrievalError(AppException):
    code = ErrorCode.RETRIEVAL_FAILED
    http_status = 500
    message = "检索失败"


# ---------------------------------------------------------------------- #
# LLM 异常
# ---------------------------------------------------------------------- #
class LLMNotConfiguredError(AppException):
    code = ErrorCode.LLM_NOT_CONFIGURED
    http_status = 503
    message = "未配置大模型 API Key, 请先在 .env 中设置 DOCMIND_LLM_API_KEY"


class LLMError(AppException):
    code = ErrorCode.LLM_ERROR
    http_status = 502
    message = "大模型调用失败"


class LLMTimeoutError(AppException):
    code = ErrorCode.LLM_TIMEOUT
    http_status = 504
    message = "大模型调用超时"


# ---------------------------------------------------------------------- #
# 语音异常
# ---------------------------------------------------------------------- #
class SpeechNotConfiguredError(AppException):
    """语音能力没配好.

    503 而不是 500: 这是"服务端缺配置", 不是"请求有问题",
    调用方重试也没用, 必须去设置里补.
    """

    code = ErrorCode.SPEECH_NOT_CONFIGURED
    http_status = 503
    message = "语音功能未配置或已关闭, 请在「设置 → 语音」里配置"


class SpeechError(AppException):
    """识别/合成过程中的失败(网络、额度、音色名写错等)."""

    code = ErrorCode.SPEECH_ERROR
    http_status = 502
    message = "语音服务调用失败"
