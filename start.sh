#!/usr/bin/env bash
# ===========================================================================
#  DocMind 一键启动 (Linux / macOS)
#
#  逻辑与 Windows 的 start.bat 完全一致, 都只是 python scripts/start.py 的启动器。
# ===========================================================================
set -e

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    if ! command -v python >/dev/null 2>&1; then
        echo
        echo "  [错误] 未找到 python3 命令"
        echo "  请先运行 ./install.sh 安装依赖"
        echo
        exit 1
    fi
    PY=python
else
    PY=python3
fi

exec "$PY" scripts/start.py "$@"
