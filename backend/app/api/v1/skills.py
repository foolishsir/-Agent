"""SKILL 接口.

给前端提供"有哪些 SKILL 可以勾选"以及"重新扫描目录"的能力.

SKILL 本身是项目内置的 Markdown 文件(见 ``skills/README.md``),
这里不存在"创建/编辑 SKILL"的接口 —— 那是有意为之:
SKILL 应该跟着代码走版本管理, 而不是变成数据库里的另一份可编辑资源.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.core.response import ok
from app.services.skills import MAX_SELECTED_SKILLS, compose_skills, get_skill_registry

router = APIRouter()


@router.get("", summary="列出可用 SKILL")
async def list_skills() -> dict[str, Any]:
    """返回可勾选的 SKILL 列表.

    **不返回正文** —— 前端只需要渲染勾选框, 而正文可能有几千字.
    需要一个 SKILL 的完整内容时走 ``GET /skills/{skill_id}``.
    """
    registry = get_skill_registry()
    skills = registry.all()

    return ok(
        {
            "items": [skill.to_public_dict() for skill in skills],
            "total": len(skills),
            "max_selected": MAX_SELECTED_SKILLS,
            "directory": str(registry.directory),
            # 加载失败的 SKILL 必须暴露: 用户写了但没出现在列表里时,
            # 没有错误信息他完全不知道哪里写错了
            "errors": registry.errors(),
        }
    )


@router.get("/{skill_id}", summary="查看 SKILL 详情(含正文)")
async def get_skill(skill_id: str) -> dict[str, Any]:
    """查看单个 SKILL 的完整内容, 便于在界面上预览它的规则。"""
    skill = get_skill_registry().get(skill_id)
    return ok(
        {
            **skill.to_public_dict(),
            "body": skill.body,
            "source": skill.source,
            "allow_finish": skill.allow_finish,
        }
    )


@router.post("/reload", summary="重新扫描 SKILL 目录")
async def reload_skills() -> dict[str, Any]:
    """重新扫描 ``skills/`` 目录, 不用重启服务.

    改完 SKILL.md 后调一次即可生效. 返回加载了几个、失败几个、失败的原因.
    """
    return ok(get_skill_registry().reload())


@router.post("/preview-compose", summary="预览多 SKILL 组合效果")
async def preview_compose(payload: dict[str, Any]) -> dict[str, Any]:
    """预览勾选多个 SKILL 时, 最终拼出来的 Prompt 与约束.

    调 SKILL 时很有用: 能直接看到"组合后追问上限变成了几层".
    也可以用来排查"为什么勾了两个 SKILL 之后风格变得四不像".
    """
    skill_ids = payload.get("skill_ids") or []
    skills = get_skill_registry().resolve([str(s) for s in skill_ids])
    prompt, constraints = compose_skills(skills)

    return ok(
        {
            "skill_ids": [s.id for s in skills],
            "prompt": prompt,
            "prompt_chars": len(prompt),
            "constraints": constraints,
        }
    )
