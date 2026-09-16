"""会话历史接口的测试.

覆盖三件事:
1. **CRUD 正确性**: 建/列/查/改/删, 以及软删的语义
2. **落库时机**: 一轮问答结束后, 用户消息与助手消息是否都写进去了
3. **隔离性**: 别人的会话必须查不到、改不了、删不掉, 且统一返回 404
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.services.conversation_service import make_title


def _h(user: str) -> dict[str, str]:
    return {"X-User-Id": user}


def _upload(client: TestClient, pdf: Path, user: str) -> str:
    return client.post(
        "/api/v1/documents",
        files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")},
        headers=_h(user),
    ).json()["data"]["document"]["id"]


# --------------------------------------------------------------------------- #
# 标题生成
# --------------------------------------------------------------------------- #
def test_title_from_question() -> None:
    assert make_title("这份文档的主要内容是什么？") == "这份文档的主要内容是什么？"


def test_title_is_truncated() -> None:
    title = make_title("问" * 100)
    assert len(title) <= 41
    assert title.endswith("…")


def test_title_strips_newlines() -> None:
    """用户粘贴多行文本提问时, 标题里不能带换行 —— 会把侧边栏布局撑坏."""
    assert "\n" not in make_title("第一行\n第二行\t第三行")


def test_title_falls_back_when_empty() -> None:
    assert make_title("   \n  ") == "新对话"


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #
def test_create_and_list_conversation(client: TestClient) -> None:
    created = client.post(
        "/api/v1/conversations", json={"title": "测试会话"}, headers=_h("u-conv-1")
    ).json()["data"]

    assert created["title"] == "测试会话"
    assert created["message_count"] == 0

    listing = client.get("/api/v1/conversations", headers=_h("u-conv-1")).json()["data"]
    assert listing["total"] == 1
    assert listing["items"][0]["id"] == created["id"]


def test_get_conversation_detail_with_messages(client: TestClient) -> None:
    conv_id = client.post("/api/v1/conversations", json={}, headers=_h("u-conv-2")).json()["data"][
        "id"
    ]

    # 未绑定文档时提问会走"没有就绪文档"的分支, 但消息依然落库 ——
    # 这正是我们要验证的: 失败/拒答的对话也应该留下记录
    client.post(
        "/api/v1/chat",
        json={"question": "测试问题", "conversation_id": conv_id},
        headers=_h("u-conv-2"),
    )

    detail = client.get(f"/api/v1/conversations/{conv_id}", headers=_h("u-conv-2")).json()["data"]
    assert len(detail["messages"]) == 2
    assert detail["messages"][0]["role"] == "user"
    assert detail["messages"][0]["content"] == "测试问题"
    assert detail["messages"][1]["role"] == "assistant"
    assert detail["messages"][1]["refused"] is True
    assert detail["message_count"] == 2


def test_first_message_sets_title(client: TestClient) -> None:
    conv_id = client.post("/api/v1/conversations", json={}, headers=_h("u-conv-3")).json()["data"][
        "id"
    ]
    assert (
        client.get(f"/api/v1/conversations/{conv_id}", headers=_h("u-conv-3")).json()["data"][
            "title"
        ]
        == "新对话"
    )

    client.post(
        "/api/v1/chat",
        json={"question": "钢刀的更换周期是多少", "conversation_id": conv_id},
        headers=_h("u-conv-3"),
    )

    title = client.get(f"/api/v1/conversations/{conv_id}", headers=_h("u-conv-3")).json()["data"][
        "title"
    ]
    assert title == "钢刀的更换周期是多少"


def test_rename_conversation(client: TestClient) -> None:
    conv_id = client.post("/api/v1/conversations", json={}, headers=_h("u-conv-4")).json()["data"][
        "id"
    ]

    renamed = client.patch(
        f"/api/v1/conversations/{conv_id}", json={"title": "新标题"}, headers=_h("u-conv-4")
    ).json()["data"]
    assert renamed["title"] == "新标题"


def test_empty_rename_rejected(client: TestClient) -> None:
    conv_id = client.post("/api/v1/conversations", json={}, headers=_h("u-conv-5")).json()["data"][
        "id"
    ]

    resp = client.patch(
        f"/api/v1/conversations/{conv_id}", json={"title": "   "}, headers=_h("u-conv-5")
    )
    assert resp.status_code == 400


def test_delete_is_soft(client: TestClient) -> None:
    """软删: 从列表消失、查不到, 但数据库里还留着(便于误删恢复与挖掘评测集)."""
    conv_id = client.post("/api/v1/conversations", json={}, headers=_h("u-conv-6")).json()["data"][
        "id"
    ]
    client.post(
        "/api/v1/chat",
        json={"question": "问一句", "conversation_id": conv_id},
        headers=_h("u-conv-6"),
    )

    result = client.delete(f"/api/v1/conversations/{conv_id}", headers=_h("u-conv-6")).json()[
        "data"
    ]
    assert result["deleted_messages"] == 2

    assert client.get(f"/api/v1/conversations/{conv_id}", headers=_h("u-conv-6")).status_code == 404
    listing = client.get("/api/v1/conversations", headers=_h("u-conv-6")).json()["data"]
    assert all(item["id"] != conv_id for item in listing["items"])


def test_clear_messages_keeps_conversation(client: TestClient) -> None:
    conv_id = client.post(
        "/api/v1/conversations", json={"title": "保留我"}, headers=_h("u-conv-7")
    ).json()["data"]["id"]
    client.post(
        "/api/v1/chat",
        json={"question": "问一句", "conversation_id": conv_id},
        headers=_h("u-conv-7"),
    )

    cleared = client.delete(
        f"/api/v1/conversations/{conv_id}/messages", headers=_h("u-conv-7")
    ).json()["data"]
    assert cleared["deleted"] == 2

    detail = client.get(f"/api/v1/conversations/{conv_id}", headers=_h("u-conv-7")).json()["data"]
    assert detail["messages"] == []
    assert detail["title"] == "保留我", "清空消息不应改标题"


def test_explicit_title_survives_first_message(client: TestClient) -> None:
    """用户显式命名的会话, 不能被第一条消息自动改名.

    这是一个真实 bug: 最初用 ``message_count == 0`` 判断"是否该自动命名",
    结果用户建了个叫"钢刀维护记录"的会话, 一问"这个多少钱"标题就变成了"这个多少钱".
    判据应该是"标题是否仍为占位符".
    """
    conv_id = client.post(
        "/api/v1/conversations", json={"title": "钢刀维护记录"}, headers=_h("u-conv-title")
    ).json()["data"]["id"]

    client.post(
        "/api/v1/chat",
        json={"question": "这个多少钱", "conversation_id": conv_id},
        headers=_h("u-conv-title"),
    )

    title = client.get(f"/api/v1/conversations/{conv_id}", headers=_h("u-conv-title")).json()[
        "data"
    ]["title"]
    assert title == "钢刀维护记录"


def test_search_by_title(client: TestClient) -> None:
    client.post("/api/v1/conversations", json={"title": "钢刀维护记录"}, headers=_h("u-conv-8"))
    client.post("/api/v1/conversations", json={"title": "锡膏储存条件"}, headers=_h("u-conv-8"))

    found = client.get(
        "/api/v1/conversations", params={"keyword": "钢刀"}, headers=_h("u-conv-8")
    ).json()["data"]
    assert found["total"] == 1
    assert "钢刀" in found["items"][0]["title"]


def test_conversations_sorted_by_recent_activity(client: TestClient) -> None:
    """最近还在用的会话应该排在最前, 而不是按创建时间."""
    first = client.post(
        "/api/v1/conversations", json={"title": "先建的"}, headers=_h("u-conv-9")
    ).json()["data"]["id"]
    second = client.post(
        "/api/v1/conversations", json={"title": "后建的"}, headers=_h("u-conv-9")
    ).json()["data"]["id"]

    # 给第一个会话发一条消息 → 它应该顶到最前面
    client.post(
        "/api/v1/chat",
        json={"question": "再问一次", "conversation_id": first},
        headers=_h("u-conv-9"),
    )

    items = client.get("/api/v1/conversations", headers=_h("u-conv-9")).json()["data"]["items"]
    assert items[0]["id"] == first
    assert items[1]["id"] == second


# --------------------------------------------------------------------------- #
# 自动建会话
# --------------------------------------------------------------------------- #
def test_create_conversation_on_first_message(client: TestClient) -> None:
    """Web 界面首次提问时后端自动建会话, 少一次往返, 也不留空会话."""
    resp = client.post(
        "/api/v1/chat",
        json={"question": "第一次提问", "create_conversation": True},
        headers=_h("u-conv-auto"),
    ).json()["data"]

    assert resp["conversation_id"]

    detail = client.get(
        f"/api/v1/conversations/{resp['conversation_id']}", headers=_h("u-conv-auto")
    ).json()["data"]
    assert detail["message_count"] == 2
    assert detail["title"] == "第一次提问"


def test_no_conversation_when_not_requested(client: TestClient) -> None:
    """不传 conversation_id 也不开自动建 → 无状态模式, 不产生任何会话记录.

    脚本和评测批跑走这条路: 它们不需要持久化, 强行落库只会在库里堆垃圾会话.
    """
    before = client.get("/api/v1/conversations", headers=_h("u-conv-stateless")).json()["data"][
        "total"
    ]

    resp = client.post(
        "/api/v1/chat", json={"question": "无状态提问"}, headers=_h("u-conv-stateless")
    ).json()["data"]

    assert resp["conversation_id"] is None
    after = client.get("/api/v1/conversations", headers=_h("u-conv-stateless")).json()["data"][
        "total"
    ]
    assert after == before


def test_conversation_doc_scope_is_reused(client: TestClient, sample_pdf: Path) -> None:
    """会话限定的文档范围在多轮之间保持一致.

    否则用户在第一轮限定了范围, 第二轮没带 doc_ids, 检索范围就悄悄变回全库了 ——
    答案的来源范围前后不一致, 非常难排查.
    """
    doc_id = _upload(client, sample_pdf, "u-conv-scope")

    conv_id = client.post(
        "/api/v1/conversations", json={"doc_ids": [doc_id]}, headers=_h("u-conv-scope")
    ).json()["data"]["id"]
    assert conv_id

    # 第二轮不传 doc_ids, 但会话记着范围, 依然能正常检索到内容
    resp = client.post(
        "/api/v1/chat",
        json={"question": "这份文档讲了什么？", "conversation_id": conv_id},
        headers=_h("u-conv-scope"),
    ).json()["data"]

    # 没有配 LLM Key 时会在生成阶段报错, 但检索阶段应该已经成功
    stages = {s["stage"] for s in resp["stages"]}
    assert "retrieval" in stages, "会话内应继承 doc_ids 并完成检索"


def test_invalid_conversation_id_rejected(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/chat",
        json={"question": "提问", "conversation_id": "not-a-real-conversation"},
        headers=_h("u-conv-bad"),
    )
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# 隔离
# --------------------------------------------------------------------------- #
def test_conversations_isolated_between_users(client: TestClient) -> None:
    conv_id = client.post(
        "/api/v1/conversations", json={"title": "私有"}, headers=_h("u-owner-c")
    ).json()["data"]["id"]

    # 别人的列表里看不到
    listing = client.get("/api/v1/conversations", headers=_h("u-thief-c")).json()["data"]
    assert all(item["id"] != conv_id for item in listing["items"])

    # 读、改、删都必须 404 —— 统一 404 而不是 403, 防止靠状态码枚举 id
    assert (
        client.get(f"/api/v1/conversations/{conv_id}", headers=_h("u-thief-c")).status_code == 404
    )
    assert (
        client.patch(
            f"/api/v1/conversations/{conv_id}", json={"title": "劫持"}, headers=_h("u-thief-c")
        ).status_code
        == 404
    )
    assert (
        client.delete(f"/api/v1/conversations/{conv_id}", headers=_h("u-thief-c")).status_code
        == 404
    )


def test_cannot_chat_into_others_conversation(client: TestClient) -> None:
    """不能往别人的会话里塞消息 —— 那等于往别人的历史记录里写入内容."""
    conv_id = client.post("/api/v1/conversations", json={}, headers=_h("u-owner-d")).json()["data"][
        "id"
    ]

    resp = client.post(
        "/api/v1/chat",
        json={"question": "注入", "conversation_id": conv_id},
        headers=_h("u-thief-d"),
    )
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #
def test_stats_counts_refusal_rate(client: TestClient) -> None:
    conv_id = client.post("/api/v1/conversations", json={}, headers=_h("u-stats")).json()["data"][
        "id"
    ]
    client.post(
        "/api/v1/chat",
        json={"question": "无文档时必然拒答", "conversation_id": conv_id},
        headers=_h("u-stats"),
    )

    stats = client.get("/api/v1/conversations/stats", headers=_h("u-stats")).json()["data"]
    assert stats["conversations"] == 1
    assert stats["messages"] == 2
    assert stats["refused"] == 1
    assert stats["refused_rate"] == 0.5
