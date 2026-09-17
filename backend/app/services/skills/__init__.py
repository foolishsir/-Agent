"""SKILL 层: 加载、查询、组合.

SKILL = 带 front-matter 的 Markdown 文件, 放在项目根的 ``skills/`` 目录下.
新增一个 SKILL 不需要改任何代码.
"""

from app.services.skills.base import Skill, parse_front_matter, parse_skill, split_front_matter
from app.services.skills.registry import (
    DEFAULT_SKILLS_DIR,
    MAX_SELECTED_SKILLS,
    SKILL_FILENAME,
    SkillRegistry,
    compose_skills,
    get_skill_registry,
    reset_skill_registry_for_test,
)

__all__ = [
    "DEFAULT_SKILLS_DIR",
    "MAX_SELECTED_SKILLS",
    "SKILL_FILENAME",
    "Skill",
    "SkillRegistry",
    "compose_skills",
    "get_skill_registry",
    "parse_front_matter",
    "parse_skill",
    "reset_skill_registry_for_test",
    "split_front_matter",
]
