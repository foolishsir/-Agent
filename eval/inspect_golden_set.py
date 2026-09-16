"""查看评测集内容(开发/人工复核用).

用法: python eval/inspect_golden_set.py [--path eval/golden_set.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GOLDEN_SET, ensure_utf8_console  # noqa: E402

ensure_utf8_console()


def main() -> int:
    parser = argparse.ArgumentParser(description="查看评测集")
    parser.add_argument("--path", default=str(GOLDEN_SET))
    parser.add_argument("--full", action="store_true", help="显示完整证据文本")
    args = parser.parse_args()

    path = Path(args.path)
    items = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    print(f"评测集: {path}  共 {len(items)} 条\n")

    for item in items:
        category = item.get("category", "?")
        tag = "无答案" if category == "no_answer" else category
        print(f"{item['id']}  [{tag}]")
        print(f"  问题: {item['question']}")

        if item.get("reference_answer"):
            print(f"  答案: {item['reference_answer'][:80]}")

        evidences = item.get("evidences") or []
        if not evidences:
            print("  证据: (无 —— 该题用于测拒答)")
        else:
            for evidence in evidences:
                text = evidence if args.full else evidence[:100]
                inside = "✓" if evidence else " "
                print(f"  证据{inside}: {text}")
        print()

    # 分布统计
    from collections import Counter

    print("-" * 60)
    print("分类分布:", dict(Counter(i.get("category") for i in items)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
