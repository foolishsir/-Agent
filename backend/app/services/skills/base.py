"""SKILL 的数据模型与解析.

SKILL 就是一个带 YAML front-matter 的 Markdown 文件:

    ---
    id: technical-interviewer
    name: 技术面试官
    max_follow_up: 3
    ---

    # 角色
    你是一位……

**两层结构是刻意的**:

- **front-matter = 结构化字段**, 由**代码**读取. 像 ``max_follow_up`` 这种参数
  要参与状态机判断(什么时候强制换话题), 如果只写在正文里让模型自律,
  就会出现"说好最多追问 3 层、实际追问 7 层"的情况. 约束必须由代码执行.
- **正文 = 给模型看的**, 原样拼进 System Prompt.

关于 front-matter 的解析
------------------------
不引入 PyYAML 作为硬依赖. 虽然它当前是被其他包顺带装上的, 但依赖一个
"碰巧存在"的包很脆 —— 换个环境可能就 ImportError. 所以这里优先用 PyYAML
(更健壮), 没有时回退到自带的简易解析器, 只覆盖 SKILL 里实际会用的子集:
字符串、整数、布尔、行内列表.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.exceptions import ParamInvalidError

#: front-matter 的分隔符
_FRONT_MATTER_DELIMITER = "---"

#: 字段名 -> (类型, 默认值)
_FIELD_SPEC: dict[str, tuple[str, Any]] = {
    "id": ("str", ""),
    "name": ("str", ""),
    "description": ("str", ""),
    "icon": ("str", "🎯"),
    "tags": ("list", []),
    "max_follow_up": ("int", 3),
    "max_turns": ("int", 25),
    "allow_finish": ("bool", True),
    "require_evidence": ("bool", True),
}

#: 必填字段
_REQUIRED_FIELDS = ("id", "name", "description")

_TRUE_VALUES = {"true", "yes", "on", "1"}
_FALSE_VALUES = {"false", "no", "off", "0"}


@dataclass
class Skill:
    """一个已加载的 SKILL."""

    id: str
    name: str
    description: str
    #: 正文(会被拼进 System Prompt)
    body: str
    icon: str = "🎯"
    tags: list[str] = field(default_factory=list)
    #: 同一话题最多追问几层. 超过后由**代码**强制换话题.
    max_follow_up: int = 3
    #: 整场最多几轮问答
    max_turns: int = 25
    allow_finish: bool = True
    #: 是否校验"问题必须溯源到简历原文"
    require_evidence: bool = True
    #: 来源文件路径(排查问题时用)
    source: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        """给前端的展示信息.

        刻意**不返回正文** —— 前端只需要渲染勾选框, 正文可能有几千字,
        每次列 SKILL 都传一遍纯属浪费带宽. 需要看正文时再单独取.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
            "tags": self.tags,
            "max_follow_up": self.max_follow_up,
            "max_turns": self.max_turns,
            "require_evidence": self.require_evidence,
        }


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #
def split_front_matter(text: str) -> tuple[str, str]:
    """把文件拆成 (front-matter 文本, 正文).

    Raises:
        ParamInvalidError: 缺少 front-matter
    """
    stripped = text.lstrip("\ufeff")  # 去掉可能的 BOM
    if not stripped.startswith(_FRONT_MATTER_DELIMITER):
        raise ParamInvalidError("SKILL.md 必须以 --- 开头的 front-matter 起始")

    # 找第二个 ---
    end = stripped.find(f"\n{_FRONT_MATTER_DELIMITER}", len(_FRONT_MATTER_DELIMITER))
    if end < 0:
        raise ParamInvalidError("SKILL.md 的 front-matter 没有结束标记 ---")

    front = stripped[len(_FRONT_MATTER_DELIMITER) : end].strip()
    body = stripped[end + len(_FRONT_MATTER_DELIMITER) + 1 :].strip()
    return front, body


def _parse_simple(front: str) -> dict[str, Any]:
    """不带 PyYAML 时的降级解析器.

    只处理 ``key: value`` 与行内列表 ``key: [a, b]`` ——
    SKILL 的 front-matter 实际只用得到这些.
    """
    data: dict[str, Any] = {}
    for line in front.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()

        # 去掉行尾注释(但不要在引号内处理 —— SKILL 里基本用不到, 保持简单)
        if "#" in value and not value.startswith(("'", '"')):
            value = value.split("#", 1)[0].strip()

        if value.startswith("[") and value.endswith("]"):
            data[key] = [
                item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()
            ]
        else:
            data[key] = value
    return data


def parse_front_matter(front: str) -> dict[str, Any]:
    """解析 front-matter, 优先用 PyYAML, 不可用时降级."""
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        return _parse_simple(front)

    try:
        parsed = yaml.safe_load(front)
    except Exception:  # noqa: BLE001 - YAML 写错了就回退, 不让它中断加载
        return _parse_simple(front)

    return parsed if isinstance(parsed, dict) else _parse_simple(front)


def _coerce(name: str, raw: Any, kind: str, default: Any) -> Any:
    """把 front-matter 里的值转成目标类型.

    为什么要显式转换: front-matter 里写 ``max_follow_up: 3``, 简易解析器
    给出的是字符串 ``"3"``, 而 PyYAML 给出的是整数 ``3``. 两条路径的结果
    必须一致 —— 否则"装了 yaml"和"没装 yaml"会出现行为差异, 这种 bug 极难排查.
    """
    if raw is None or raw == "":
        return default

    if kind == "int":
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise ParamInvalidError(f"字段 {name} 必须是整数, 实际是 {raw!r}") from exc

    if kind == "bool":
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in _TRUE_VALUES:
            return True
        if text in _FALSE_VALUES:
            return False
        raise ParamInvalidError(f"字段 {name} 必须是布尔值, 实际是 {raw!r}")

    if kind == "list":
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        return [item.strip() for item in re.split(r"[,，]", str(raw)) if item.strip()]

    return str(raw).strip()


def parse_skill(text: str, *, source: str = "") -> Skill:
    """把 SKILL.md 的内容解析成 Skill 对象.

    Raises:
        ParamInvalidError: 缺少必需字段或字段类型不对
    """
    front, body = split_front_matter(text)
    raw = parse_front_matter(front)

    missing = [name for name in _REQUIRED_FIELDS if not str(raw.get(name, "")).strip()]
    if missing:
        raise ParamInvalidError(f"缺少必需字段: {', '.join(missing)}")

    if not body.strip():
        raise ParamInvalidError("SKILL 正文为空 —— 至少要有角色与提问规则")

    values = {
        name: _coerce(name, raw.get(name), kind, default)
        for name, (kind, default) in _FIELD_SPEC.items()
    }

    # 上限校验: 数值不合理时直接拒绝, 而不是让它跑起来后把候选人问崩
    if not 1 <= values["max_follow_up"] <= 10:
        raise ParamInvalidError(f"max_follow_up 应在 1~10 之间, 实际是 {values['max_follow_up']}")
    if not 3 <= values["max_turns"] <= 100:
        raise ParamInvalidError(f"max_turns 应在 3~100 之间, 实际是 {values['max_turns']}")

    return Skill(body=body, source=source, **values)
