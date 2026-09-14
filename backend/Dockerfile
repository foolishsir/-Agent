# =============================================================================
# DocMind 后端镜像
#
# 构建:  docker build -t docmind-backend:0.1.0 ./backend
# 说明:  采用多阶段构建; 依赖层单独缓存, 改业务代码时不会重新装依赖.
# =============================================================================

# ------------------------------ 构建阶段 ------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先只复制依赖清单, 最大化利用 Docker 层缓存
COPY requirements.txt .

# torch 的 CPU 版比默认版小 10 倍以上(默认版会拉 CUDA 运行时约 2.5GB).
# 容器里通常没有 GPU, 这一步能把镜像从 ~5GB 压到 ~1.2GB, 是部署环节最大的优化点.
RUN pip install --prefix=/install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install --prefix=/install -r requirements.txt \
        -i https://pypi.tuna.tsinghua.edu.cn/simple

# ------------------------------ 运行阶段 ------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai \
    # 容器内固定走镜像下载模型; 生产环境建议改为挂载预下载好的模型目录
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/app/models

WORKDIR /app

# 运行期只需 libgomp(torch 依赖的 OpenMP 运行时)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . /app

# 以非 root 用户运行: 容器逃逸时的最小权限原则
RUN useradd --create-home --shell /bin/bash docmind \
    && mkdir -p /app/data /app/logs /app/models \
    && chown -R docmind:docmind /app
USER docmind

EXPOSE 8000

# 健康检查直接复用就绪探针的业务语义
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

# 生产不用 --reload, worker 数按 CPU 核数调整
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
