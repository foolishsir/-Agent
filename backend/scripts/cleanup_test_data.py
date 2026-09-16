"""清理开发库里的测试残留数据.

为什么需要这个脚本
------------------
测试曾经因为 **conftest 漏设 ``DOCMIND_DATABASE_URL``** 而直接写进开发库 ——
config.py 会把 sqlite 的相对路径解析到项目根目录, 于是 ``data/docmind.db``
被测试数据灌满(实测 370+ 份文档). 这个 bug 已经修掉(见 conftest),
但之前产生的脏数据需要清一次.

这个脚本同时也是"存储泄漏"的体检工具: 它会交叉比对三处存储
(关系库 / 向量库 / 磁盘文件), 找出孤儿数据并清理.

用法::

    # 先看会删什么(不实际删除)
    python backend/scripts/cleanup_test_data.py --dry-run

    # 实际删除测试用户的数据 + 全库孤儿数据
    python backend/scripts/cleanup_test_data.py

    # 只清孤儿(不动任何真实用户的数据)
    python backend/scripts/cleanup_test_data.py --orphans-only
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from contextlib import suppress
from pathlib import Path

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        with suppress(Exception):
            _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import settings  # noqa: E402

#: 测试用例使用的 user_id 前缀/名单. 这些是测试里 hardcode 的,
#: 真实用户不会用这种形式的名字.
TEST_USER_PREFIXES = ("u-",)
TEST_USER_EXACT = {"demo", "zhangyuke", "test", "testuser"}


def is_test_user(user_id: str) -> bool:
    return user_id.startswith(TEST_USER_PREFIXES) or user_id in TEST_USER_EXACT


def db_path_from_url(url: str) -> Path:
    """从 SQLAlchemy 连接串里取出 sqlite 文件路径."""
    _, _, raw = url.partition(":///")
    return Path(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="清理开发库中的测试残留数据")
    parser.add_argument("--dry-run", action="store_true", help="只统计, 不删除")
    parser.add_argument("--orphans-only", action="store_true", help="只清孤儿数据, 不动测试用户")
    args = parser.parse_args()

    db_path = db_path_from_url(settings.database_url)
    if not db_path.exists():
        print(f"[FAIL] 数据库不存在: {db_path}")
        return 1

    print(f"数据库: {db_path}")
    print(f"上传目录: {settings.upload_dir}")
    print(f"向量库: {settings.chroma_persist_dir}")
    print(f"模式: {'DRY-RUN(不删除)' if args.dry_run else '实际删除'}\n")

    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys=ON")
    cur = con.cursor()

    # ---------------- 1. 找出测试用户与孤儿文档 ----------------
    all_docs = {
        row[0]: (row[1], row[2])  # doc_id -> (user_id, file_path)
        for row in cur.execute("SELECT id, user_id, file_path FROM documents")
    }

    test_doc_ids = [doc_id for doc_id, (user_id, _) in all_docs.items() if is_test_user(user_id)]

    keep_doc_ids = set(all_docs) - set(test_doc_ids)

    # 孤儿文档: 状态是 DELETED 但没有对应的向量/文件(纯残留)
    orphan_doc_ids = [
        doc_id
        for doc_id in keep_doc_ids
        if cur.execute("SELECT status FROM documents WHERE id = ?", (doc_id,)).fetchone()[0]
        == "DELETED"
    ]

    targets = orphan_doc_ids if args.orphans_only else test_doc_ids

    print("=== 待清理的文档 ===")
    print(f"  数据库文档总数    : {len(all_docs)}")
    print(f"  测试用户文档      : {len(test_doc_ids)}")
    print(f"  已软删的孤儿文档  : {len(orphan_doc_ids)}")
    print(f"  本次将清理        : {len(targets)}")
    print(f"  保留              : {len(all_docs) - len(targets)}")

    chunk_count = 0
    if targets:
        placeholders = ",".join("?" * len(targets))
        chunk_count = cur.execute(
            f"SELECT COUNT(*) FROM chunks WHERE doc_id IN ({placeholders})", targets
        ).fetchone()[0]
    print(f"  关联分块          : {chunk_count}")

    # ---------------- 2. 会话 ----------------
    conv_ids = [
        row[0]
        for row in cur.execute("SELECT id, user_id FROM conversations")
        if not args.orphans_only and is_test_user(row[1])
    ]
    msg_count = 0
    if conv_ids:
        placeholders = ",".join("?" * len(conv_ids))
        msg_count = cur.execute(
            f"SELECT COUNT(*) FROM messages WHERE conversation_id IN ({placeholders})", conv_ids
        ).fetchone()[0]
    print(f"\n=== 待清理的会话 ===\n  会话 {len(conv_ids)} 个, 消息 {msg_count} 条")

    # ---------------- 3. 磁盘文件孤儿 ----------------
    referenced = {Path(path).name for _, path in all_docs.values() if path}
    disk_files = {p.name: p for p in settings.upload_dir.glob("*") if p.is_file()}
    orphan_files = [
        p
        for name, p in disk_files.items()
        if name not in referenced and not name.startswith(".upload_")
    ]
    print(f"\n=== 磁盘 ===\n  上传目录文件 {len(disk_files)} 个, 其中孤儿 {len(orphan_files)} 个")

    # ---------------- 4. 向量库孤儿 ----------------
    orphan_vectors = []
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        collection = client.get_or_create_collection(
            name=settings.chroma_collection, metadata={"hnsw:space": "cosine"}
        )
        stored: list[str] = []
        offset = 0
        # 必须翻页: Chroma 的 get() 单次返回有条数上限, 一次拿全会**静默截断**,
        # 导致大批孤儿向量被漏掉 —— 而"清理脚本自己漏数据"是最不该发生的事.
        while True:
            page = collection.get(include=[], limit=500, offset=offset).get("ids", [])
            stored.extend(page)
            if len(page) < 500:
                break
            offset += 500

        for chunk_id in stored:
            # chunk id 形如 {doc_id}_p0001_c000, 取前缀即 doc_id
            doc_id = chunk_id.split("_p")[0]
            if doc_id not in all_docs or doc_id in set(targets):
                orphan_vectors.append(chunk_id)
        print(f"\n=== 向量库 ===\n  向量总数 {len(stored)}, 其中孤儿/待清理 {len(orphan_vectors)}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n[WARN] 向量库检查失败(可能服务正在运行): {exc}")

    if args.dry_run:
        print("\n[DRY-RUN] 未做任何修改. 去掉 --dry-run 才会真正删除.")
        con.close()
        return 0

    # ---------------- 执行清理 ----------------
    print("\n=== 开始清理 ===")

    if targets:
        placeholders = ",".join("?" * len(targets))
        cur.execute(f"DELETE FROM chunks WHERE doc_id IN ({placeholders})", targets)
        cur.execute(f"DELETE FROM documents WHERE id IN ({placeholders})", targets)
        print(f"  已删除 {len(targets)} 份文档及其分块")

    if conv_ids:
        placeholders = ",".join("?" * len(conv_ids))
        cur.execute(f"DELETE FROM messages WHERE conversation_id IN ({placeholders})", conv_ids)
        cur.execute(f"DELETE FROM conversations WHERE id IN ({placeholders})", conv_ids)
        print(f"  已删除 {len(conv_ids)} 个会话及其消息")

    con.commit()

    # 磁盘
    removed_files = 0
    for path in orphan_files:
        with suppress(OSError):
            path.unlink()
            removed_files += 1
    # 测试文档对应的原始文件.
    # 先 exists() 再删: missing_ok=True 对不存在的文件不报错, 若直接计数会把
    # "已经不在的文件"也算成"已删除", 输出里就会出现明显夸大的数字.
    for doc_id in targets:
        file_path = Path(all_docs[doc_id][1])
        if file_path.exists():
            with suppress(OSError):
                file_path.unlink()
                removed_files += 1
    print(f"  已删除 {removed_files} 个文件")

    # 向量库
    if orphan_vectors:
        try:
            for start in range(0, len(orphan_vectors), 500):
                collection.delete(ids=orphan_vectors[start : start + 500])
            print(f"  已删除 {len(orphan_vectors)} 条向量")
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] 向量删除失败: {exc}")

    con.execute("VACUUM")
    con.close()

    print("\n清理完成. 建议重启服务后再访问页面.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
