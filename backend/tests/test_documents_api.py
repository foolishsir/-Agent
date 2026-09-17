"""文档接口端到端测试.

跑的是**真实链路**: 上传真实 PDF 字节流 → 解析 → 分块 → 假向量化 → 写入真实 Chroma.
只有 Embedding 模型被替换成确定性假实现(见 conftest), 其余全部是真的 ——
包括向量库的写入与元数据过滤.

每个用例用**不同的 X-User-Id**, 避免会话级 client 带来的用例间状态污染.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.services.vectorstore import SearchFilter, get_vector_store


def _upload(client: TestClient, pdf: Path, user: str = "u-upload"):
    return client.post(
        "/api/v1/documents",
        files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")},
        headers={"X-User-Id": user},
    )


# --------------------------------------------------------------------------- #
# 上传与入库
# --------------------------------------------------------------------------- #
def test_upload_pdf_completes_ingest(client: TestClient, sample_pdf: Path) -> None:
    """上传后 inline 模式会同步跑完入库, 返回时已经是 READY."""
    resp = _upload(client, sample_pdf, user="u-ready")

    assert resp.status_code == 201
    body = resp.json()
    assert body["code"] == "OK"

    doc = body["data"]["document"]
    assert body["data"]["created"] is True
    assert doc["status"] == "READY"
    assert doc["page_count"] == 3
    assert doc["char_count"] > 200
    assert doc["parent_chunk_count"] > 0
    assert doc["child_chunk_count"] > 0
    # 写入向量库的条数必须等于子块数 —— 少一条就说明有分块没被索引
    assert doc["chunks_indexed"] == doc["child_chunk_count"]


def test_upload_records_timing_for_observability(client: TestClient, sample_pdf: Path) -> None:
    """解析与向量化耗时必须落库, 否则无法定位性能瓶颈在哪一环."""
    doc = _upload(client, sample_pdf, user="u-timing").json()["data"]["document"]
    assert doc["parse_cost_ms"] >= 0
    assert doc["embed_cost_ms"] >= 0


def test_chunks_are_actually_written_to_vector_store(client: TestClient, sample_pdf: Path) -> None:
    """端到端确认: 向量库里真的能查到这份文档的分块."""
    doc = _upload(client, sample_pdf, user="u-vector").json()["data"]["document"]

    store = get_vector_store()
    hits = store.list_by_doc(doc["id"])

    assert len(hits) == doc["child_chunk_count"]
    for hit in hits:
        assert hit.metadata["doc_id"] == doc["id"]
        assert hit.metadata["user_id"] == "u-vector"
        assert hit.metadata["chunk_type"] == "child"
        assert hit.metadata["page_start"] >= 1
        assert hit.document  # 文本也存进去了, BM25 检索要用


def test_metadata_has_no_none_values(client: TestClient, sample_pdf: Path) -> None:
    """Chroma 的 metadata 不接受 None, 写入边界必须清洗干净."""
    doc = _upload(client, sample_pdf, user="u-meta").json()["data"]["document"]
    for hit in get_vector_store().list_by_doc(doc["id"]):
        assert all(value is not None for value in hit.metadata.values())


# --------------------------------------------------------------------------- #
# 幂等
# --------------------------------------------------------------------------- #
def test_reupload_same_file_is_idempotent(client: TestClient, sample_pdf: Path) -> None:
    """同一份文件重复上传必须复用, 而不是在向量库里存两份."""
    first = _upload(client, sample_pdf, user="u-idem").json()["data"]
    second = _upload(client, sample_pdf, user="u-idem").json()["data"]

    assert second["created"] is False
    assert second["document"]["id"] == first["document"]["id"]

    # 向量库里的条数没有翻倍
    assert (
        len(get_vector_store().list_by_doc(first["document"]["id"]))
        == (first["document"]["child_chunk_count"])
    )


def test_idempotency_is_per_user(client: TestClient, sample_pdf: Path) -> None:
    """不同用户上传同一份文件应该各自拥有一份, 不能互相复用."""
    a = _upload(client, sample_pdf, user="u-alice").json()["data"]
    b = _upload(client, sample_pdf, user="u-bob").json()["data"]

    assert a["document"]["id"] != b["document"]["id"]


# --------------------------------------------------------------------------- #
# 查询
# --------------------------------------------------------------------------- #
def test_list_documents(client: TestClient, sample_pdf: Path) -> None:
    _upload(client, sample_pdf, user="u-list")
    resp = client.get("/api/v1/documents", headers={"X-User-Id": "u-list"})

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["total"] >= 1
    assert data["page"] == 1
    assert any(item["status"] == "READY" for item in data["items"])


def test_list_can_filter_by_status(client: TestClient, sample_pdf: Path) -> None:
    _upload(client, sample_pdf, user="u-filter")
    resp = client.get(
        "/api/v1/documents", params={"status": "READY"}, headers={"X-User-Id": "u-filter"}
    )
    assert all(item["status"] == "READY" for item in resp.json()["data"]["items"])


def test_get_document_detail_and_status(client: TestClient, sample_pdf: Path) -> None:
    doc_id = _upload(client, sample_pdf, user="u-detail").json()["data"]["document"]["id"]

    detail = client.get(f"/api/v1/documents/{doc_id}", headers={"X-User-Id": "u-detail"}).json()
    assert detail["data"]["id"] == doc_id

    status = client.get(
        f"/api/v1/documents/{doc_id}/status", headers={"X-User-Id": "u-detail"}
    ).json()["data"]
    assert status["status"] == "READY"
    assert status["progress"] == 100


def test_missing_document_returns_404(client: TestClient) -> None:
    resp = client.get("/api/v1/documents/does-not-exist", headers={"X-User-Id": "u-x"})
    assert resp.status_code == 404
    assert resp.json()["code"] == "NOT_FOUND"


# --------------------------------------------------------------------------- #
# 多用户隔离
# --------------------------------------------------------------------------- #
def test_documents_are_isolated_between_users(client: TestClient, sample_pdf: Path) -> None:
    doc_id = _upload(client, sample_pdf, user="u-owner").json()["data"]["document"]["id"]

    # 换个用户就查不到 —— 而且返回 404 而不是 403:
    # 如果对"无权限"返回 403, 攻击者就能靠状态码差异枚举出系统里有哪些 doc_id
    resp = client.get(f"/api/v1/documents/{doc_id}", headers={"X-User-Id": "u-intruder"})
    assert resp.status_code == 404

    listing = client.get("/api/v1/documents", headers={"X-User-Id": "u-intruder"}).json()
    assert all(item["id"] != doc_id for item in listing["data"]["items"])


def test_vector_filter_isolates_users(client: TestClient, sample_pdf: Path) -> None:
    """向量库层面的隔离: 过滤条件必须真的生效, 而不是靠应用层事后筛."""
    _upload(client, sample_pdf, user="u-vec-a")
    _upload(client, sample_pdf, user="u-vec-b")

    store = get_vector_store()
    count_a = store.count(SearchFilter(user_id="u-vec-a"))
    count_b = store.count(SearchFilter(user_id="u-vec-b"))

    assert count_a > 0
    assert count_b > 0
    assert store.count(SearchFilter(user_id="u-nobody")) == 0


def test_empty_doc_filter_matches_nothing(client: TestClient) -> None:
    """空 doc_ids 表示"不允许匹配任何文档", 而不是"不加限制".

    这是一个容易写错的边界: 如果实现成"空列表 = 不过滤",
    那么用户在没有任何文档时提问, 会检索到**全库**的内容 —— 严重的越权泄露.
    """
    store = get_vector_store()
    assert store.count(SearchFilter(doc_ids=[])) == 0


# --------------------------------------------------------------------------- #
# 文件校验
# --------------------------------------------------------------------------- #
def test_non_pdf_extension_rejected(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/documents",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
        headers={"X-User-Id": "u-bad-ext"},
    )
    assert resp.status_code == 415
    assert resp.json()["code"] == "UNSUPPORTED_FILE_TYPE"


def test_fake_pdf_with_wrong_magic_rejected(client: TestClient) -> None:
    """扩展名可以随便改, 所以必须校验文件头魔数."""
    resp = client.post(
        "/api/v1/documents",
        files={"file": ("fake.pdf", b"PK\x03\x04 this is actually a zip", "application/pdf")},
        headers={"X-User-Id": "u-bad-magic"},
    )
    assert resp.status_code == 415
    assert resp.json()["code"] == "UNSUPPORTED_FILE_TYPE"


def test_empty_file_rejected(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/documents",
        files={"file": ("empty.pdf", b"", "application/pdf")},
        headers={"X-User-Id": "u-empty"},
    )
    assert resp.status_code == 415


# --------------------------------------------------------------------------- #
# 删除
# --------------------------------------------------------------------------- #
def test_delete_removes_chunks_and_vectors(client: TestClient, sample_pdf: Path) -> None:
    doc_id = _upload(client, sample_pdf, user="u-del").json()["data"]["document"]["id"]
    store = get_vector_store()
    assert store.count(SearchFilter(doc_ids=[doc_id])) > 0

    resp = client.delete(f"/api/v1/documents/{doc_id}", headers={"X-User-Id": "u-del"})
    assert resp.status_code == 200

    data = resp.json()["data"]
    assert data["doc_id"] == doc_id
    assert data["deleted_vectors"] > 0
    assert data["deleted_chunks"] > 0
    assert data["file_removed"] is True

    # 关系库与向量库都必须查不到 —— 否则就是"幽灵数据"
    assert store.count(SearchFilter(doc_ids=[doc_id])) == 0
    assert (
        client.get(f"/api/v1/documents/{doc_id}", headers={"X-User-Id": "u-del"}).status_code == 404
    )


def test_deleted_document_not_in_list(client: TestClient, sample_pdf: Path) -> None:
    doc_id = _upload(client, sample_pdf, user="u-dellist").json()["data"]["document"]["id"]
    client.delete(f"/api/v1/documents/{doc_id}", headers={"X-User-Id": "u-dellist"})

    listing = client.get("/api/v1/documents", headers={"X-User-Id": "u-dellist"}).json()
    assert all(item["id"] != doc_id for item in listing["data"]["items"])


def test_reupload_after_delete_is_reprocessed(client: TestClient, sample_pdf: Path) -> None:
    """删除后重新上传同一份文件应该被当成新任务处理, 而不是返回已删除的记录."""
    first = _upload(client, sample_pdf, user="u-redo").json()["data"]
    client.delete(f"/api/v1/documents/{first['document']['id']}", headers={"X-User-Id": "u-redo"})

    second = _upload(client, sample_pdf, user="u-redo").json()["data"]
    assert second["document"]["status"] == "READY"
    assert get_vector_store().count(SearchFilter(doc_ids=[second["document"]["id"]])) > 0


def test_cannot_delete_other_users_document(client: TestClient, sample_pdf: Path) -> None:
    doc_id = _upload(client, sample_pdf, user="u-real-owner").json()["data"]["document"]["id"]

    resp = client.delete(f"/api/v1/documents/{doc_id}", headers={"X-User-Id": "u-thief"})
    assert resp.status_code == 404
    # 原主人的文档依然在
    assert (
        client.get(f"/api/v1/documents/{doc_id}", headers={"X-User-Id": "u-real-owner"}).status_code
        == 200
    )


def test_can_delete_every_document_until_list_is_empty(
    client: TestClient, sample_pdf: Path, tmp_root: Path
) -> None:
    """**没有"必须保留一个文档"的限制** —— 可以一路删到 0.

    这条是为一个真实误解加的: 用户删不掉文档, 以为系统要求至少留一份。
    实际原因是前端把 `busy` 状态在渲染时烙进了按钮的 disabled,
    上传过一次之后删除按钮就永久失效了(详见 frontend-tests/check-dom-ids.js 的说明)。

    后端这条契约本身是正确的, 但**没有任何测试盯着它** ——
    万一将来有人"顺手"加一条"至少保留一个"的保护(比如为了 demo 好看),
    就会悄悄破坏用户的预期。所以在这里钉死。

    注意两份文档必须**内容不同**: 内容相同会命中 MD5 幂等, 复用同一条记录,
    那样只落到一份文档上, 测不到"删多份"。
    """
    import pymupdf as fitz

    user = "u-delete-all"
    ids: list[str] = []
    for marker in ("A", "B", "C"):
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"Unique delete-test document {marker}", fontsize=14)
        payload = doc.tobytes()
        doc.close()

        resp = client.post(
            "/api/v1/documents",
            files={"file": (f"del-{marker}.pdf", payload, "application/pdf")},
            headers={"X-User-Id": user},
        )
        assert resp.status_code in (200, 201), resp.text
        ids.append(resp.json()["data"]["document"]["id"])

    listing = client.get("/api/v1/documents", headers={"X-User-Id": user}).json()["data"]
    assert len(listing["items"]) == 3

    # 逐个删, 每次都应该真的少一个
    for index, doc_id in enumerate(ids):
        resp = client.delete(f"/api/v1/documents/{doc_id}", headers={"X-User-Id": user})
        assert resp.status_code == 200, resp.text
        remaining = client.get("/api/v1/documents", headers={"X-User-Id": user}).json()["data"]
        expected = 3 - index - 1
        assert len(remaining["items"]) == expected, (
            f"删了第 {index + 1} 份后应该剩 {expected} 份, 实际 {len(remaining['items'])}"
        )

    # 删到 0 —— 列表为空, 而不是报错或被拒
    final = client.get("/api/v1/documents", headers={"X-User-Id": user}).json()["data"]
    assert final["items"] == []
    assert final["total"] == 0


# --------------------------------------------------------------------------- #
# 重新处理
# --------------------------------------------------------------------------- #
def test_reindex_is_idempotent(client: TestClient, sample_pdf: Path) -> None:
    """重跑入库不能让向量库里的条数翻倍 —— chunk id 确定性生成的意义就在这里."""
    doc = _upload(client, sample_pdf, user="u-reindex").json()["data"]["document"]
    original_count = doc["child_chunk_count"]

    resp = client.post(f"/api/v1/documents/{doc['id']}/reindex", headers={"X-User-Id": "u-reindex"})
    assert resp.status_code == 200

    store = get_vector_store()
    assert store.count(SearchFilter(doc_ids=[doc["id"]])) == original_count
