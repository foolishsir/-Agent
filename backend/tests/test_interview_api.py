"""面试链路的**集成测试**（用桩模型，不调真实 API）。

为什么要有这一层
----------------
``test_interview.py`` 测的是纯函数（决策表、溯源校验）。
但整条链路还有一类问题只有跑通接口才会暴露：

- 状态在前后端之间传递时丢字段（``follow_up_depth`` / ``topic_index`` 没还原）
- 提纲解析失败时整个面试起不来
- 追问层级没有随决策正确递增，导致追问永远停不下来
- 复盘报告在模型返回非 JSON 时崩掉

这些都不是"某个函数写错了"，而是**装配错了**。用桩模型跑一遍，
既能覆盖装配，又不依赖网络和 API Key（Key 过期时也能跑）。

桩模型模拟的是一个"听话但死板"的模型：
规划提纲时返回固定提纲，评估时按回答长度判断深浅，提问时返回固定格式。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.services.llm.base import ChatMessage, ChatResult

# --------------------------------------------------------------------------- #
# 桩模型
# --------------------------------------------------------------------------- #
OUTLINE = [
    {
        "topic": "Redis 缓存",
        "angle": "追问 40% 的测量口径",
        "opening": "你说的响应时间提升 40% 是怎么测的?",
    },
    {"topic": "父子分块", "angle": "追问选型理由", "opening": "为什么选父子分块而不是固定长度?"},
    {"topic": "自建评测集", "angle": "追问指标定义", "opening": "MRR 0.988 是在什么数据上算的?"},
]


class StubLLM:
    """按调用内容识别当前是"规划 / 评估 / 提问 / 复盘"哪一步，返回对应结果。"""

    configured = True
    name = "stub:test"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def achat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        system = messages[0].content
        user = messages[-1].content
        self.calls.append({"system": system[:20], "user_chars": len(user)})

        # ① 提纲规划
        if "面试提纲规划助手" in system:
            return ChatResult(content=json.dumps(OUTLINE, ensure_ascii=False))

        # ② 回答评估
        if "面试评估员" in system:
            answer = user.split("【回答】")[-1]
            shallow = len(answer.strip()) < 60
            payload = {
                "depth": "shallow" if shallow else "deep",
                "specificity": 0.3 if shallow else 0.9,
                "has_numbers": not shallow,
                "has_tradeoff": not shallow,
                "vague_words": ["大概", "差不多"] if shallow else [],
                "highlight": "" if shallow else "说清了 5 分钟过期的取舍",
                "doubt": "没说测量口径" if shallow else "",
            }
            return ChatResult(content=json.dumps(payload, ensure_ascii=False))

        # ④ 复盘报告
        if "复盘报告" in system:
            return ChatResult(
                content=json.dumps(
                    {
                        "overall": "整体表达清楚，但量化结果的来源说得不够扎实。",
                        "highlights": [
                            {"point": "取舍意识", "evidence": "提到了 5 分钟过期的折中"}
                        ],
                        "concerns": [{"point": "数字口径", "evidence": "40% 未说明基线"}],
                        "suggestions": ["补齐测量方法", "准备被否决方案的复盘"],
                        "score": {"technical_depth": 6, "expression": 7, "authenticity": 7},
                    },
                    ensure_ascii=False,
                )
            )

        # ③ 提问：带一段模型爱加的前缀，验证代码层会剥掉
        return ChatResult(content="好的，那么请具体讲讲你在这部分做了什么?")


@pytest.fixture
def stub_llm(monkeypatch: pytest.MonkeyPatch) -> StubLLM:
    """把 conductor 与 interview API 里的 get_llm_client 都换成桩。"""
    from app.api.v1 import interview as interview_api
    from app.services.interview import conductor

    stub = StubLLM()
    monkeypatch.setattr(conductor, "get_llm_client", lambda: stub)
    monkeypatch.setattr(interview_api, "get_llm_client", lambda: stub)
    return stub


# --------------------------------------------------------------------------- #
# 用例
# --------------------------------------------------------------------------- #
RESUME_NAME = "resume-for-interview.pdf"
RESUME_TEXT = b"""Zhang Ming
Backend engineer. Used Redis cache to improve order query by 40 percent.
Built a RAG project with parent child chunking and Chroma vector store.
Self built evaluation set with 46 items, MRR reached 0.988.
"""


def _make_resume_pdf() -> bytes:
    import pymupdf as fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Resume", fontsize=16)
    y = 140
    for line in RESUME_TEXT.decode().splitlines():
        page.insert_text((72, y), line, fontsize=10)
        y += 16
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def ready_doc(client) -> str:
    """上传一份可预测的简历并等它 READY。"""
    res = client.post(
        "/api/v1/documents",
        files={"file": (RESUME_NAME, _make_resume_pdf(), "application/pdf")},
        headers={"X-User-Id": "interview-test"},
    )
    assert res.status_code == 201, res.text
    return res.json()["data"]["document"]["id"]


def _body(doc_id: str, **extra: Any) -> dict[str, Any]:
    return {"doc_id": doc_id, "skill_ids": ["technical-interviewer"], **extra}


def test_skills_are_listed_for_the_frontend(client):
    """前端勾选框的数据源: 不带正文, 但必须带约束字段。

    ``max_follow_up`` / ``max_turns`` 不在列表里的话, 前端勾选框上就显示不出
    "追问 ≤ 3 层", 用户根本不知道勾了这个 SKILL 会发生什么。
    """
    res = client.get("/api/v1/skills")
    assert res.status_code == 200
    data = res.json()["data"]

    assert data["total"] >= 2
    assert data["max_selected"] == 2
    assert data["errors"] == []

    for item in data["items"]:
        assert item["id"] and item["name"]
        for field in ("max_follow_up", "max_turns", "require_evidence"):
            assert field in item
        # 列表接口不返回正文 —— 正文几千字, 每次列一遍纯属浪费
        assert "body" not in item


def test_reload_is_idempotent(client):
    """「重新扫描」按钮连点两次结果应该一致, 不能出现重复 SKILL。"""
    first = client.post("/api/v1/skills/reload").json()["data"]
    second = client.post("/api/v1/skills/reload").json()["data"]
    assert first["loaded"] == second["loaded"] > 0
    assert second["failed"] == 0

    listed = client.get("/api/v1/skills").json()["data"]
    assert listed["total"] == second["loaded"]


def test_start_interview_returns_outline_and_first_question(client, ready_doc, stub_llm):
    res = client.post("/api/v1/interview/start", json=_body(ready_doc))
    assert res.status_code == 200, res.text
    data = res.json()["data"]

    # 提纲: 面试要有主线, 而不是每轮临时想一个话题
    assert len(data["outline"]) == len(OUTLINE)
    assert data["outline"][0]["topic"] == "Redis 缓存"

    # 问题: 桩模型故意加了"好的，那么"前缀, 代码层必须剥掉
    assert data["question"].startswith("请具体讲讲")
    assert "好的" not in data["question"][:4]

    assert data["decision"] == "OPEN"
    assert data["constraints"]["max_follow_up"] >= 1
    assert data["resume_chars"] > 0
    assert data["resume_truncated"] is False


def test_shallow_answer_triggers_follow_up(client, ready_doc, stub_llm):
    """答得浅 → 追问, 且追问层级 +1。"""
    start = client.post("/api/v1/interview/start", json=_body(ready_doc)).json()["data"]

    res = client.post(
        "/api/v1/interview/next",
        json=_body(
            ready_doc,
            outline=start["outline"],
            turns=[{"question": start["question"], "answer": "用了 Redis, 大概快了一些。"}],
            follow_up_depth=start["follow_up_depth"],
            topic_index=start["topic_index"],
        ),
    )
    assert res.status_code == 200, res.text
    data = res.json()["data"]

    assert data["decision"] == "FOLLOW_UP"
    assert data["follow_up_depth"] == start["follow_up_depth"] + 1
    # 追问时话题不该前进
    assert data["topic_index"] == start["topic_index"]
    assert data["evaluation"]["vague_words"]
    assert "模糊表述" in data["evaluation_hint"]


def test_deep_answer_moves_to_next_topic(client, ready_doc, stub_llm):
    """答得具体且有取舍 → 换话题, 且追问层级归零。

    "不要为了难而难" —— 把已经答对的人继续往死里问, 得到的是噪声不是信号。
    """
    start = client.post("/api/v1/interview/start", json=_body(ready_doc)).json()["data"]

    deep = (
        "基线是上线前一周的 P95，800ms。我先加了联合索引把范围查询走成索引，"
        "降到 300ms；再对近七天热点订单做 Redis 缓存，key 是订单 ID，"
        "过期 5 分钟，最终 120ms。选 5 分钟是因为订单状态会变更，"
        "过期太长会读到旧状态，我们评估过一致性要求，这是可接受的折中。"
    )
    res = client.post(
        "/api/v1/interview/next",
        json=_body(
            ready_doc,
            outline=start["outline"],
            turns=[{"question": start["question"], "answer": deep}],
            follow_up_depth=1,
            topic_index=start["topic_index"],
        ),
    )
    data = res.json()["data"]

    assert data["decision"] == "NEXT_TOPIC"
    assert data["follow_up_depth"] == 0
    assert data["topic_index"] == start["topic_index"] + 1


def test_follow_up_stops_at_limit(client, ready_doc, stub_llm):
    """追问层级到顶后必须换话题 —— 硬约束不能被"回答很浅"覆盖。

    这是整个链路里最该锁死的一条: 如果把判断交给模型, 它一定会超。
    """
    start = client.post("/api/v1/interview/start", json=_body(ready_doc)).json()["data"]
    max_follow_up = start["constraints"]["max_follow_up"]

    res = client.post(
        "/api/v1/interview/next",
        json=_body(
            ready_doc,
            outline=start["outline"],
            turns=[{"question": start["question"], "answer": "不清楚。"}],
            follow_up_depth=max_follow_up,  # 已经追到上限
            topic_index=start["topic_index"],
        ),
    )
    data = res.json()["data"]
    assert data["decision"] == "NEXT_TOPIC"


def test_turn_limit_finishes_interview(client, ready_doc, stub_llm):
    """轮次到上限 → finished, 且不再生成问题。"""
    start = client.post("/api/v1/interview/start", json=_body(ready_doc)).json()["data"]
    max_turns = start["constraints"]["max_turns"]

    turns = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(max_turns)]
    res = client.post(
        "/api/v1/interview/next",
        json=_body(ready_doc, outline=start["outline"], turns=turns),
    )
    data = res.json()["data"]

    assert data["finished"] is True
    assert data["decision"] == "FINISH"
    assert data["question"] == ""
    assert str(max_turns) in data["reason"]


def test_summary_returns_structured_report(client, ready_doc, stub_llm):
    res = client.post(
        "/api/v1/interview/summary",
        json=_body(
            ready_doc,
            turns=[
                {"question": "40% 怎么测的?", "answer": "大概是压测出来的。"},
                {"question": "为什么用 5 分钟过期?", "answer": "考虑了一致性和性能的折中。"},
            ],
        ),
    )
    assert res.status_code == 200, res.text
    report = res.json()["data"]["report"]

    assert report["score"]["technical_depth"] == 6
    assert report["highlights"][0]["point"] == "取舍意识"
    assert report["concerns"][0]["evidence"] == "40% 未说明基线"
    assert res.json()["data"]["turn_count"] == 2


def test_summary_rejects_empty_interview(client, ready_doc, stub_llm):
    res = client.post("/api/v1/interview/summary", json=_body(ready_doc, turns=[]))
    assert res.status_code == 400
    assert "没有任何对话" in res.json()["message"]


def test_unknown_document_is_rejected(client, stub_llm):
    res = client.post("/api/v1/interview/start", json=_body("no-such-doc"))
    assert res.status_code == 400
    assert "文档不存在" in res.json()["message"]


def test_missing_doc_id_is_rejected(client, stub_llm):
    res = client.post("/api/v1/interview/start", json={"skill_ids": []})
    assert res.status_code == 400
    assert "doc_id" in res.json()["message"]


def test_unknown_skill_id_is_rejected(client, ready_doc, stub_llm):
    """勾了一个不存在的 SKILL 必须报错, 而不是静默用默认风格跑完。

    静默降级在这里是有害的: 用户以为自己在用某个 SKILL,
    实际跑的是默认风格, 却没有任何提示。
    """
    res = client.post(
        "/api/v1/interview/start",
        json={"doc_id": ready_doc, "skill_ids": ["not-a-real-skill"]},
    )
    # 404 而不是 400: 这是"资源不存在", 不是"参数格式不对"
    assert res.status_code == 404
    message = res.json()["message"]
    assert "not-a-real-skill" in message
    # 报错信息里要列出可用的 SKILL, 否则用户不知道正确的 id 是什么
    assert "technical-interviewer" in message


def test_too_many_skills_is_rejected(client, ready_doc, stub_llm):
    """最多勾 2 个 —— 钩子多了面试风格会互相稀释。

    注意这条在**去重之前**判断: 传 [a, b, a] 应该被拒绝,
    而不是"去重后剩 2 个所以放行"。
    """
    res = client.post(
        "/api/v1/interview/start",
        json={
            "doc_id": ready_doc,
            "skill_ids": ["technical-interviewer", "project-deep-dive", "technical-interviewer"],
        },
    )
    assert res.status_code == 400
    assert "最多同时启用 2 个" in res.json()["message"]


def test_no_skill_still_works(client, ready_doc, stub_llm):
    """一个 SKILL 都不勾也应该能面 —— 只是没有额外风格约束。"""
    res = client.post("/api/v1/interview/start", json={"doc_id": ready_doc, "skill_ids": []})
    assert res.status_code == 200
    assert res.json()["data"]["skill_ids"] == []
