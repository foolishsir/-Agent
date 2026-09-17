"""PDF 解析质量检查脚本 —— 在写问答链路之前先把文档链路调对.

用法::

    python backend/scripts/parse_pdf.py "D:/path/to/文档.pdf"
    python backend/scripts/parse_pdf.py 文档.pdf --show-chunks 5 --show-text 3

为什么需要它
------------
RAG 的效果上限由**文档链路**决定: 解析漏了、分块切碎了, 后面 Prompt 写得再好也救不回来.
有了这个脚本, 换一份新 PDF 时可以先花 10 秒看解析结果对不对,
而不用启动整个服务、上传、再提问, 从答案反推问题出在哪.
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import suppress
from pathlib import Path

# Windows 控制台 GBK 编码处理 —— 与 check_env.py 同一套处理
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        with suppress(Exception):
            _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import REPO_ROOT, bootstrap  # noqa: E402

# 加载界面配置(data/runtime_settings.json).
# 不加载的话脚本会拿到 .env 与代码默认值, 而不是用户在界面上改的值 ——
# 表现是"我明明配了 Key, 脚本说没配"这种误导性结论.
bootstrap(quiet=True)

PROJECT_ROOT = REPO_ROOT

from app.services.chunking import chunk_document  # noqa: E402
from app.services.parser import clean_document, parse_pdf  # noqa: E402

RULE = "=" * 78


def section(title: str) -> None:
    print(f"\n{RULE}\n {title}\n{RULE}")


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 PDF 解析与分块质量")
    parser.add_argument("pdf", help="PDF 文件路径")
    parser.add_argument("--show-text", type=int, default=5, help="展示前 N 个清洗后的段落")
    parser.add_argument("--show-chunks", type=int, default=3, help="展示前 N 个子块")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"[FAIL] 文件不存在: {pdf_path}")
        return 1

    # ---------------- 解析 ----------------
    section("1. 解析(PyMuPDF + 坐标 + 分栏 + 页眉页脚)")
    started = time.perf_counter()
    parsed = parse_pdf(pdf_path)
    parse_ms = (time.perf_counter() - started) * 1000

    print(f"文件      : {parsed.filename}")
    print(f"页数      : {parsed.page_count}")
    print(f"字符数    : {parsed.char_count}")
    print(f"耗时      : {parse_ms:.0f} ms")
    print(f"疑似扫描件: {'是(需要 OCR)' if parsed.is_scanned else '否'}")
    if parsed.metadata.get("title"):
        print(f"PDF 标题  : {parsed.metadata['title']}")

    heading_blocks = sum(1 for p in parsed.pages for b in p.blocks if b.is_heading)
    print(f"识别标题数: {heading_blocks}")

    for page in parsed.pages[:3]:
        print(
            f"\n--- 第 {page.page_no} 页: {len(page.blocks)} 个文本块, {page.text_length} 字符 ---"
        )
        for block in page.blocks[:3]:
            preview = block.text.replace("\n", " / ")[:90]
            mark = "[标题]" if block.is_heading else "      "
            print(
                f"  {mark} x={block.x0:6.1f} y={block.y0:6.1f} "
                f"字号={block.font_size:4.1f} | {preview}"
            )

    # ---------------- 清洗 ----------------
    section("2. 清洗(去页眉页脚已在上一步完成, 这里做断行合并与去噪)")
    cleaned = clean_document(parsed)
    print(f"段落数    : {len(cleaned.paragraphs)}")
    print(f"有效字符  : {cleaned.char_count}")
    print(f"标题段落  : {sum(1 for p in cleaned.paragraphs if p.is_heading)}")

    if args.show_text:
        print(f"\n--- 前 {args.show_text} 个段落 ---")
        for paragraph in cleaned.paragraphs[: args.show_text]:
            mark = "[标题]" if paragraph.is_heading else "      "
            preview = paragraph.text[:150].replace("\n", " ")
            print(f"  {mark} P{paragraph.page_no} ({paragraph.char_count}字) {preview}")

    # ---------------- 分块 ----------------
    section("3. 父子块切分")
    started = time.perf_counter()
    chunking = chunk_document(cleaned, doc_id="DEMO0000")
    chunk_ms = (time.perf_counter() - started) * 1000

    children = chunking.children
    parents = chunking.parents
    print(f"父块数    : {len(parents)}")
    print(f"子块数    : {len(children)}")
    print(f"耗时      : {chunk_ms:.0f} ms")
    if children:
        sizes = [c.char_count for c in children]
        print(f"子块长度  : min={min(sizes)} max={max(sizes)} avg={sum(sizes) / len(sizes):.0f}")
        # 压缩比: 子块总字符 / 全文有效字符. 远大于 1 说明重叠或重复过多
        total_child_chars = sum(sizes)
        print(
            f"字符膨胀率: {total_child_chars / max(cleaned.char_count, 1):.2f}x "
            "(含重叠; 明显 >1.5 说明 overlap 配置偏大)"
        )

    if args.show_chunks and children:
        print(f"\n--- 前 {args.show_chunks} 个子块 ---")
        for child in children[: args.show_chunks]:
            print(f"\n  id={child.id}  页码=P{child.page_start}-{child.page_end}")
            print(f"  章节={child.section_path or '(无)'}")
            print(f"  正文={child.content[:200]}")
            print(f"  送入模型的文本(含章节前缀)={child.embedding_text[:120]}...")

    # ---------------- 健康度告警 ----------------
    section("4. 质量预警")
    warnings: list[str] = []
    if parsed.is_scanned:
        warnings.append("文档疑似扫描件 —— 需要 OCR, 当前版本会直接拒绝处理")
    if cleaned.char_count < 200:
        warnings.append("有效字符过少 —— 可能解析失败, 或文档本身几乎没有文字")
    if children:
        average = sum(c.char_count for c in children) / len(children)
        if average < 80:
            warnings.append(
                f"子块平均长度仅 {average:.0f} 字 —— 检索容易命中半句话, "
                "建议调大 DOCMIND_CHILD_CHUNK_SIZE"
            )
        if average > 600:
            warnings.append(
                f"子块平均长度 {average:.0f} 字偏大 —— 语义容易被稀释, "
                "建议调小 DOCMIND_CHILD_CHUNK_SIZE"
            )
        empty_sections = sum(1 for c in children if not c.section_path)
        if empty_sections / len(children) > 0.6:
            warnings.append(
                f"{empty_sections}/{len(children)} 个子块没有章节路径 —— "
                "标题识别可能失效(检查是否字号规则未命中)"
            )
    if not children:
        warnings.append("没有产生任何分块 —— 文档链路存在严重问题")

    if warnings:
        for item in warnings:
            print(f"[WARN] {item}")
    else:
        print("[ OK ] 未发现明显问题")

    print("\n提示: 解析与分块质量直接决定 RAG 效果上限. 换新文档时先跑一遍本脚本.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
