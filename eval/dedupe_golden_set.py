"""评测集去重工具.

用法: python eval/dedupe_golden_set.py [--path eval/golden_set.jsonl] [--dry-run]

为什么要单独一个脚本: 生成器里已经有去重, 但已经生成的旧评测集也需要清理一遍.
把去重做成可重复执行的工具, 比"重新生成一遍"更省 LLM 调用.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GOLDEN_SET, ensure_utf8_console, save_golden_set  # noqa: E402

ensure_utf8_console()


def main() -> int:
    parser = argparse.ArgumentParser(description="评测集去重")
    parser.add_argument("--path", default=str(GOLDEN_SET))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from eval.metrics import GoldenItem, normalize

    path = Path(args.path)
    raw = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    items = [
        GoldenItem(
            id=str(r["id"]),
            question=str(r["question"]),
            reference_answer=str(r.get("reference_answer", "")),
            evidences=[str(e) for e in r.get("evidences", [])],
            category=str(r.get("category", "semantic")),
            source_pages=[int(p) for p in r.get("source_pages", [])],
            doc_ids=[str(d) for d in r.get("doc_ids", [])],
        )
        for r in raw
    ]

    seen: set[str] = set()
    unique: list[GoldenItem] = []
    duplicates: list[str] = []
    for item in items:
        key = normalize(item.question)
        if key in seen:
            duplicates.append(item.question)
            continue
        seen.add(key)
        unique.append(item)

    print(f"共 {len(items)} 条, 重复 {len(duplicates)} 条, 去重后 {len(unique)} 条")
    for question in duplicates:
        print(f"  [重复] {question[:60]}")

    if not duplicates:
        print("没有重复项")
        return 0

    if args.dry_run:
        print("\n[DRY-RUN] 未写入")
        return 0

    for i, item in enumerate(unique, start=1):
        item.id = f"q{i:03d}"
    save_golden_set(unique, path)
    print(f"\n已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
