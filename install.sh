#!/usr/bin/env bash
# ===========================================================================
#  DocMind 一键安装 (Linux / macOS)
#
#  逻辑与 Windows 的 install.bat 完全一致 —— 都只是 python scripts/setup.py
#  的启动器。放在这里是为了让非 Windows 用户也有同样的体验。
# ===========================================================================
set -e

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    if ! command -v python >/dev/null 2>&1; then
        echo
        echo "  [错误] 未找到 python3 命令"
        echo "  请先安装 Python 3.11+: https://www.python.org/downloads/"
        echo
        exit 1
    fi
    PY=python
else
    PY=python3
fi

exec "$PY" scripts/setup.py
