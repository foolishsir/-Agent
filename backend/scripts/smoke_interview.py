"""一次性冒烟脚本: 上传示例简历 → 开始面试 → 回答两轮 → 生成复盘.

跑完即可删除. 目的是验证面试链路端到端能通, 而不是靠"接口返回 200"。
"""

from __future__ import annotations

import json
import pathlib
import urllib.request
import uuid

BASE = "http://127.0.0.1:8000/api/v1"
HDR = {"X-User-Id": "demo-user"}
PDF = pathlib.Path(__file__).resolve().parents[2] / "samples" / "张明-后端开发-示例简历.pdf"


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={**HDR, "Content-Type": "application/json"},
        method="POST",
    )
    return json.loads(urllib.request.urlopen(req, timeout=300).read())["data"]


def upload(path: pathlib.Path) -> dict:
    b = uuid.uuid4().hex
    body = b"".join(
        [
            f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="resume.pdf"\r\n'
            f"Content-Type: application/pdf\r\n\r\n".encode(),
            path.read_bytes(),
            f"\r\n--{b}--\r\n".encode(),
        ]
    )
    req = urllib.request.Request(
        BASE + "/documents",
        data=body,
        headers={**HDR, "Content-Type": f"multipart/form-data; boundary={b}"},
        method="POST",
    )
    return json.loads(urllib.request.urlopen(req, timeout=300).read())["data"]


def main() -> None:
    doc = upload(PDF)["document"]
    print(f"[1] 上传完成 doc_id={doc['id']} status={doc['status']}")
    print(
        f"    {doc['page_count']} 页 / {doc['char_count']} 字 / "
        f"父块 {doc['parent_chunk_count']} 子块 {doc['child_chunk_count']} / "
        f"解析 {doc['parse_cost_ms']}ms 向量化 {doc['embed_cost_ms']}ms"
    )

    start = post(
        "/interview/start",
        {"doc_id": doc["id"], "skill_ids": ["technical-interviewer", "project-deep-dive"]},
    )
    print(f"\n[2] 面试开始 | 提纲 {len(start['outline'])} 个点 | 约束 {start['constraints']}")
    for i, t in enumerate(start["outline"], 1):
        print(f"    {i}. {t['topic']} —— {t['angle']}")
    print(f"\n    第 1 问: {start['question']}")
    print(f"    可溯源: {start['traceability']}")

    # 模拟一个"答得比较浅"的回答 —— 应该触发追问
    shallow = "我用了 Redis 做缓存，就是把热点数据放到 Redis 里，然后查询的时候先从 Redis 查。"
    r2 = post(
        "/interview/next",
        {
            "doc_id": doc["id"],
            "skill_ids": ["technical-interviewer", "project-deep-dive"],
            "outline": start["outline"],
            "turns": [{"question": start["question"], "answer": shallow}],
            "follow_up_depth": start["follow_up_depth"],
            "topic_index": start["topic_index"],
        },
    )
    print(f"\n[3] 第一轮回答(刻意答浅) → 决策={r2['decision']}")
    print(f"    评估: {r2['evaluation']}")
    print(f"    评价: {r2['evaluation_hint']}")
    print(f"    第 2 问: {r2['question']}")

    # 第二轮给一个"具体、有取舍"的回答 —— 应该换话题
    deep = (
        "订单查询接口原来 P95 是 800ms，瓶颈在订单表全表扫描。我先加了 (user_id, status, "
        "create_time) 的联合索引把范围查询走成索引，P95 降到 300ms；再对近 7 天的热点订单做 "
        "Redis 缓存，key 是订单 ID，过期时间 5 分钟，最终 P95 是 120ms。40% 这个数字是拿 "
        "上线前后各一周的监控 P95 对比算的。之所以用 5 分钟而不是更长，是因为订单状态会变更，"
        "过期太长会出现读到旧状态的问题，我们评估过一致性要求，5 分钟是可以接受的折中。"
    )
    turns = [
        {"question": start["question"], "answer": shallow},
        {"question": r2["question"], "answer": deep},
    ]
    r3 = post(
        "/interview/next",
        {
            "doc_id": doc["id"],
            "skill_ids": ["technical-interviewer", "project-deep-dive"],
            "outline": start["outline"],
            "turns": turns,
            "follow_up_depth": r2["follow_up_depth"],
            "topic_index": r2["topic_index"],
        },
    )
    print(f"\n[4] 第二轮回答(具体+有取舍) → 决策={r3['decision']}")
    print(f"    评估: {r3['evaluation']}")
    print(f"    第 3 问: {r3['question']}")

    turns.append(
        {"question": r3["question"], "answer": "这块我了解得不多，主要是靠文档和同事帮忙。"}
    )
    rep = post(
        "/interview/summary",
        {
            "doc_id": doc["id"],
            "skill_ids": ["technical-interviewer", "project-deep-dive"],
            "turns": turns,
        },
    )
    print(f"\n[5] 复盘报告 ({rep['turn_count']} 轮):")
    print(json.dumps(rep["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
