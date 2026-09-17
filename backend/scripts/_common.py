"""独立脚本的公共引导.

**为什么需要这个文件**
--------------------
项目支持"在网页上改配置"(存到 ``data/runtime_settings.json``),
但这些配置**只有在应用启动时才被加载** —— 由 ``main.py`` 的 lifespan
调用 ``load_runtime_overrides()``.

于是所有直接调 service 层的独立脚本都读不到界面配置, 只能拿到
``.env`` 与代码默认值. 踩到的实例:

    用户在界面上配好了阿里云百炼 Key, 服务端 ``/speech/status`` 显示可用,
    但 ``smoke_speech.py`` 报"未配置百炼 Key" —— **工具给了误导性的结论**,
    比不报还糟: 会让人去查一个根本不存在的问题.

同类风险:
- ``parse_pdf.py`` / ``bench_large_pdf.py``: 会用默认分块参数而不是界面调的
- ``check_env.py``: 会报"未配置 LLM Key", 哪怕界面上配好了

**唯一正确的做法是让脚本走和服务器一样的加载路径**, 而不是让每个脚本
自己读 json 文件 —— 那样迟早会漏掉字段或读错优先级.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"

_bootstrapped = False


def bootstrap(*, quiet: bool = False) -> int:
    """把后端加入 import 路径, 并加载界面上的运行时配置.

    必须在**导入任何 app.* 模块之前**调用: 配置对象在模块导入时就被实例化了,
    晚一步就晚了 —— 这是 conftest.py 里同样强调过的顺序问题.

    Returns:
        实际应用了多少项界面配置(0 表示没有配置文件或全部为空).
    """
    global _bootstrapped

    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))

    if _bootstrapped:
        return 0

    from app.services.config_service import load_runtime_overrides

    applied = load_runtime_overrides()
    _bootstrapped = True

    if not quiet:
        if applied:
            print(f"[配置] 已加载 {applied} 项界面配置（data/runtime_settings.json）")
        else:
            print("[配置] 未发现界面配置, 使用 .env 与代码默认值")
    return applied


__all__ = ["BACKEND_DIR", "REPO_ROOT", "bootstrap"]
