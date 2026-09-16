"""全局配置中心.

设计要点
--------
1. 所有配置项均支持环境变量覆盖, 环境变量统一使用 ``DOCMIND_`` 前缀,
   避免与系统已有的 ``DEBUG`` / ``HOST`` / ``PORT`` 等通用变量冲突.
2. 使用 ``pydantic-settings`` 做类型校验: 配置写错在进程启动时就报错,
   而不是等到运行到某行代码才抛异常.
3. 通过 ``get_settings()`` 做进程级单例缓存, 避免每次请求重复解析 .env.
4. 路径类配置统一在 ``model_validator`` 中解析为基于项目根目录的绝对路径,
   保证无论从哪个工作目录启动服务, 落盘位置都一致.
5. 列表类配置使用 ``Annotated[list[str], NoDecode]``.
   这是一个必须知道的坑: pydantic-settings 对复杂类型(list/dict)默认会先尝试
   ``json.loads`` 解析环境变量. 若 .env 里写 ``.pdf,.docx`` 这种逗号分隔形式,
   会在**校验器执行之前**就抛 SettingsError. 加 NoDecode 关掉自动解码,
   改由下面的 field_validator 自己处理, 才能支持更符合直觉的逗号分隔写法.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# backend/app/core/config.py -> docmind/
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

ENV_FILE: Path = PROJECT_ROOT / ".env"

# 把 .env 也灌进 os.environ.
# 原因: 部分第三方库(如 huggingface_hub 读取的 HF_ENDPOINT)只看进程环境变量,
# 不经过 pydantic-settings. 不加载就会出现「.env 里明明配了镜像却依然超时」的坑.
# override=False 保证真实的系统环境变量优先级更高, 便于容器化部署时覆盖.
load_dotenv(ENV_FILE, override=False)


class Settings(BaseSettings):
    """DocMind 全部运行时配置."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="DOCMIND_",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # 应用基础
    # ------------------------------------------------------------------ #
    app_name: str = "DocMind"
    app_version: str = "0.1.0"
    app_env: Literal["dev", "test", "prod"] = "dev"
    debug: bool = True
    api_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )

    # ------------------------------------------------------------------ #
    # LLM (OpenAI 兼容协议, 默认对接 DeepSeek)
    # ------------------------------------------------------------------ #
    llm_provider: str = "deepseek"
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_timeout: float = 60.0
    llm_max_retries: int = 2

    # ------------------------------------------------------------------ #
    # Embedding
    # ------------------------------------------------------------------ #
    embedding_provider: Literal["local", "openai"] = "local"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    embedding_max_length: int = 512
    # BGE 中文系列官方建议: query 侧加指令前缀, passage 侧不加.
    # 这是一个极易踩坑的细节 —— 两侧不一致会显著拉低召回率.
    embedding_query_instruction: str = "为这个句子生成表示以用于检索相关文章："
    # HuggingFace 镜像, 国内网络建议设置为 https://hf-mirror.com
    hf_endpoint: str = ""

    # ------------------------------------------------------------------ #
    # Rerank 精排
    # ------------------------------------------------------------------ #
    rerank_enabled: bool = True
    rerank_provider: Literal["local", "none"] = "local"
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_device: str = "cpu"
    rerank_top_n: int = 5
    # 精排分数低于该阈值时判定为「文档中无相关内容」, 直接拒答而非交给 LLM 编造.
    rerank_min_score: float = 0.0

    # ------------------------------------------------------------------ #
    # 分块策略
    # ------------------------------------------------------------------ #
    # parent_child = 章节感知的父子块(推荐)
    # recursive    = 按分隔符优先级递归切分
    # fixed        = 固定长度硬切(对比实验用的基线)
    chunk_strategy: str = "parent_child"
    parent_chunk_size: int = 1500
    child_chunk_size: int = 300
    chunk_overlap: int = 50
    #: 过短的分块会被并入相邻块, 避免产生无信息量的向量
    min_chunk_size: int = 30
    #: 递归切分的分隔符优先级, 用 | 分隔(见 chunking/params.py 的说明)
    chunk_separators: str = "\\n\\n|\\n|。|！|？|；|…|，"
    #: 是否把章节标题保留在子块正文里.
    #: False(默认) 时正文保持干净, 靠 embedding_text 拼章节路径来补语境
    chunk_keep_heading: bool = False

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    vector_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    final_top_k: int = 5

    # ------------------------------------------------------------------ #
    # 存储
    # ------------------------------------------------------------------ #
    data_dir: Path = Path("data")
    upload_dir: Path = Path("data/uploads")
    log_dir: Path = Path("logs")

    # 向量库: embedded = 进程内嵌 Chroma(零依赖, 单机首选)
    #         http     = 连接独立 Chroma Server(可水平扩展)
    chroma_mode: Literal["embedded", "http"] = "embedded"
    chroma_persist_dir: Path = Path("data/chroma")
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "docmind_chunks"

    database_url: str = "sqlite+aiosqlite:///./data/docmind.db"
    redis_url: str = "redis://localhost:6379/0"

    # 任务执行模式: inline = 请求内同步执行(开发调试, 无需 Redis)
    #               queue  = 投递到 Redis 队列由 Worker 异步消费
    task_mode: Literal["inline", "queue"] = "inline"
    task_max_retries: int = 2

    # ------------------------------------------------------------------ #
    # 上传限制
    # ------------------------------------------------------------------ #
    max_upload_size_mb: int = 50
    allowed_extensions: Annotated[list[str], NoDecode] = Field(default_factory=lambda: [".pdf"])
    # 文本抽取字符数低于该值时, 判定为扫描件, 触发 OCR 降级链路
    ocr_fallback_threshold: int = 80

    # 启动时是否预热本地模型.
    # 预热把"模型加载"的开销从第一个用户请求移到进程启动阶段,
    # 代价是启动变慢(本地 BGE 在 CPU 上约 3~10 秒). 测试环境下应关闭.
    warmup_on_startup: bool = True

    # 是否允许通过 Web 界面修改运行时配置(LLM Key / 检索参数等).
    #
    # 默认开启, 是为了让首次使用者不必去编辑 .env 文件就能把项目跑起来.
    # 但**线上部署必须关掉**, 或在前置网关加鉴权 ——
    # 该接口没有身份校验, 任何能访问它的人都能改写配置(含 API Key).
    # 这是"开箱易用 vs 安全默认"的显式取舍, 不是疏忽.
    allow_runtime_config: bool = True

    # ================================================================== #
    # 校验与派生属性
    # ================================================================== #
    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def _normalize_extensions(cls, v: object) -> object:
        """支持 .env 中写成 ``.pdf,.docx`` 的逗号分隔形式."""
        if isinstance(v, str):
            return [item.strip().lower() for item in v.split(",") if item.strip()]
        if isinstance(v, list):
            return [str(item).strip().lower() for item in v]
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _normalize_origins(cls, v: object) -> object:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @model_validator(mode="after")
    def _resolve_paths(self) -> Settings:
        """把相对路径统一解析成基于项目根目录的绝对路径."""
        for name in ("data_dir", "upload_dir", "log_dir", "chroma_persist_dir"):
            raw: Path = getattr(self, name)
            if not raw.is_absolute():
                raw = PROJECT_ROOT / raw
            setattr(self, name, raw.resolve())

        # SQLite 连接串里的相对路径同样要锚定到项目根目录.
        # 否则从不同工作目录启动服务会连到两个不同的数据库文件,
        # 表现为"本地开发有数据, 部署后数据全没了".
        if self.database_url.startswith("sqlite"):
            scheme, sep, raw_path = self.database_url.partition(":///")
            if sep and raw_path:
                db_path = Path(raw_path)
                if not db_path.is_absolute():
                    db_path = (PROJECT_ROOT / raw_path).resolve()
                # SQLAlchemy 在 Windows 上要求绝对路径用正斜杠形式
                self.database_url = f"{scheme}:///{db_path.as_posix()}"

        return self

    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"

    @property
    def llm_configured(self) -> bool:
        """LLM 是否已配置可用(未配置时问答链路应给出明确提示而非 500)."""
        return bool(self.llm_api_key.strip())

    def ensure_dirs(self) -> None:
        """确保所有落盘目录存在(幂等)."""
        for directory in (self.data_dir, self.upload_dir, self.log_dir, self.chroma_persist_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置单例."""
    return Settings()


settings: Settings = get_settings()
