"""SKILL 注册表 —— 扫描目录、加载、校验、组合.

设计约定
--------
**项目内置、不进数据库、不支持在网页上编写.** 新增一个 SKILL 只需要往
``skills/`` 目录里放一个 ``SKILL.md``, **不需要改任何代码** ——
这样"加一种面试风格"的成本从"改后端 + 改前端"降到"写一个 Markdown 文件".

为什么不做成数据库里的可编辑资源:
- SKILL 是**随代码一起演进**的资产, 应该进版本管理、走 code review
- 网页编辑器意味着要处理富文本、版本、权限、XSS, 复杂度远超收益
- 而且 SKILL 写错了会直接影响面试质量, 让它跟着代码走更容易追溯

加载策略是"**失败隔离**": 某个 SKILL 写错了只跳过它并记录原因,
不影响其它 SKILL, 也不会让服务起不来.
"""

from __future__ import annotations

import threading
from pathlib import Path

from app.core.exceptions import NotFoundError, ParamInvalidError
from app.core.logging import get_logger, log_kv
from app.services.skills.base import Skill, parse_skill

logger = get_logger("docmind.skills")

#: SKILL 文件名(固定)
SKILL_FILENAME = "SKILL.md"

#: 项目根目录 / skills
DEFAULT_SKILLS_DIR = Path(__file__).resolve().parents[4] / "skills"

#: 一次最多勾选几个 SKILL.
#: 限制成 2 个是刻意的: 3 个以上风格会互相冲突 ——
#: 一个说要深挖三层、另一个说要快速覆盖, 模型会给出四不像的面试.
MAX_SELECTED_SKILLS = 2

#: 不勾任何 SKILL 时的兜底约束.
#:
#: 有兜底值很重要: 如果"没勾 SKILL"等于"没有约束", 那追问会永远停不下来、
#: 轮次也没有上限 —— 一个沉默的默认值比一个显式的默认值危险得多.
DEFAULT_MAX_FOLLOW_UP = 3
DEFAULT_MAX_TURNS = 20

#: 不勾任何 SKILL 时的中性风格说明.
#: 刻意写得短 —— 它的作用只是"别让模型自由发挥", 而不是替用户定义面试风格.
DEFAULT_INTERVIEW_PROMPT = """你是一位技术面试官。

- 只问候选人简历里出现过的内容。不确定某个词是否在简历里, 就不要问。
- 每次只问一个问题, 不要做评价、不要给正确答案。
- 回答停留在"用过/了解"层面就追问实现细节; 回答里有数字就追问测量口径;
  回答里有技术选型就追问被放弃的方案。
- 回答已经具体到能说清取舍时, 换下一个话题 —— 不要为了难而难。
- 只输出问题本身, 不要任何前缀、铺垫或过渡。"""


class SkillRegistry:
    """扫描目录并维护可用 SKILL 的注册表."""

    def __init__(self, directory: Path | None = None) -> None:
        self._dir = directory or DEFAULT_SKILLS_DIR
        self._skills: dict[str, Skill] = {}
        self._errors: list[dict[str, str]] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # 加载
    # ------------------------------------------------------------------ #
    def reload(self) -> dict[str, object]:
        """重新扫描目录. 返回加载结果摘要(供 /skills/reload 接口使用)."""
        skills: dict[str, Skill] = {}
        errors: list[dict[str, str]] = []

        if not self._dir.exists():
            logger.warning("SKILL 目录不存在 | path=%s", self._dir)
            with self._lock:
                self._skills, self._errors = {}, []
            return {"loaded": 0, "failed": 0, "directory": str(self._dir), "exists": False}

        for path in sorted(self._dir.glob(f"*/{SKILL_FILENAME}")):
            try:
                text = path.read_text(encoding="utf-8")
                skill = parse_skill(text, source=str(path.relative_to(self._dir.parent)))
            except Exception as exc:  # noqa: BLE001 - 单个 SKILL 出错不能影响其它
                # 失败隔离: 记下来继续加载下一个. 一个写错的 SKILL 不该让
                # 整个功能不可用, 但必须让用户能看见错在哪.
                errors.append({"path": str(path), "error": str(exc)})
                logger.warning("SKILL 加载失败, 已跳过 | path=%s error=%s", path, exc)
                continue

            if skill.id in skills:
                errors.append({"path": str(path), "error": f"id 重复: {skill.id}"})
                logger.warning("SKILL id 重复, 已跳过 | id=%s path=%s", skill.id, path)
                continue

            skills[skill.id] = skill

        with self._lock:
            self._skills = skills
            self._errors = errors

        log_kv(
            logger,
            "skills.loaded",
            loaded=len(skills),
            failed=len(errors),
            ids=",".join(sorted(skills)),
        )
        return {
            "loaded": len(skills),
            "failed": len(errors),
            "directory": str(self._dir),
            "exists": True,
            "errors": errors,
        }

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    @property
    def directory(self) -> Path:
        return self._dir

    def all(self) -> list[Skill]:
        """全部可用 SKILL, 按名称排序."""
        with self._lock:
            return sorted(self._skills.values(), key=lambda s: s.name)

    def get(self, skill_id: str) -> Skill:
        with self._lock:
            skill = self._skills.get(skill_id)
        if skill is None:
            available = ", ".join(sorted(self._skills)) or "(无)"
            raise NotFoundError(f"SKILL 不存在: {skill_id}. 可用: {available}")
        return skill

    def resolve(self, skill_ids: list[str]) -> list[Skill]:
        """把前端传来的 id 列表解析成 Skill 对象, 并做数量校验.

        数量校验发生在**去重之前** —— 传 ``[a, b, a]`` 这种带重复的列表
        应该直接拒绝, 而不是"去重后剩 2 个所以放行".
        前端的勾选框本来就不可能产生重复项, 出现重复说明调用方有问题,
        与其猜他的意图, 不如报错。
        """
        if not skill_ids:
            return []

        if len(skill_ids) > MAX_SELECTED_SKILLS:
            # 用 ParamInvalidError 而不是 NotFoundError:
            # "选多了"是请求参数不对(400), 不是"资源找不到"(404)。
            # 返回 404 会让调用方以为是 SKILL 不存在, 排查方向完全跑偏。
            raise ParamInvalidError(
                f"最多同时启用 {MAX_SELECTED_SKILLS} 个 SKILL, 传了 {len(skill_ids)} 个"
            )

        # 去重但保持顺序 —— 第一个是"首要 SKILL", 顺序会影响 Prompt 结构
        seen: set[str] = set()
        ordered: list[str] = []
        for skill_id in skill_ids:
            if skill_id not in seen:
                seen.add(skill_id)
                ordered.append(skill_id)

        return [self.get(skill_id) for skill_id in ordered]

    def errors(self) -> list[dict[str, str]]:
        """上次加载失败的 SKILL.

        必须暴露出来 —— 用户写了个 SKILL 但没出现在列表里时,
        如果没有错误信息, 他完全不知道是哪里写错了.
        """
        with self._lock:
            return list(self._errors)


# --------------------------------------------------------------------------- #
# 组合
# --------------------------------------------------------------------------- #
def compose_skills(skills: list[Skill]) -> tuple[str, dict[str, object]]:
    """把多个 SKILL 组合成一段 Prompt, 并算出合并后的约束.

    Returns:
        ``(prompt 文本, 合并后的约束)``

    组合规则
    --------
    - **正文**: 第一个是首要(完整), 后续的标注为"补充视角"
    - **结构化字段取更严格的一方**:
      ``max_follow_up`` / ``max_turns`` 取 min, ``require_evidence`` 任一为 true 则 true

    为什么要取最严: 这和"限流参数取最小值""权限取交集"是同一个思路 ——
    **安全相关的约束在组合时只能更严, 不能更松**. 如果取 max,
    用户勾一个宽松的 SKILL 就能绕过另一个的追问上限.
    """
    if not skills:
        # 一个 SKILL 都不勾是合法状态 —— 用户可能只想用系统的默认面试风格。
        # 这时给一份中性的风格说明和宽松的兜底约束:
        # 追问上限默认 3 层(和内置 SKILL 一致), 轮次上限 20,
        # 不然 "0 个 SKILL" 会变成"可以无限追问无限轮次", 那是很糟的默认值。
        return DEFAULT_INTERVIEW_PROMPT, {
            "max_follow_up": DEFAULT_MAX_FOLLOW_UP,
            "max_turns": DEFAULT_MAX_TURNS,
            "require_evidence": False,
            "allow_finish": True,
        }

    parts: list[str] = []
    for index, skill in enumerate(skills):
        header = "【首要面试风格】" if index == 0 else f"【补充视角 {index}】"
        parts.append(f"{header}\n{skill.body}")

    prompt = "\n\n---\n\n".join(parts)

    constraints: dict[str, object] = {
        "max_follow_up": min(s.max_follow_up for s in skills),
        "max_turns": min(s.max_turns for s in skills),
        "require_evidence": any(s.require_evidence for s in skills),
        "allow_finish": all(s.allow_finish for s in skills),
    }
    return prompt, constraints


# --------------------------------------------------------------------------- #
# 单例
# --------------------------------------------------------------------------- #
_registry: SkillRegistry | None = None
_registry_lock = threading.Lock()


def get_skill_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                registry = SkillRegistry()
                registry.reload()
                _registry = registry
    return _registry


def reset_skill_registry_for_test() -> None:
    global _registry
    with _registry_lock:
        _registry = None


__all__ = [
    "DEFAULT_INTERVIEW_PROMPT",
    "DEFAULT_MAX_FOLLOW_UP",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_SKILLS_DIR",
    "MAX_SELECTED_SKILLS",
    "SKILL_FILENAME",
    "SkillRegistry",
    "compose_skills",
    "get_skill_registry",
    "reset_skill_registry_for_test",
]
