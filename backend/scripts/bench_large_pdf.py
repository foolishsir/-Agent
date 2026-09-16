"""大文档吞吐压测 —— 摸清"多少页会慢到什么程度".

为什么要先测再优化
------------------
"大文件处理慢"是一句没有信息量的判断. 真正要回答的是:
**慢在解析、分块还是向量化? 是线性增长还是有拐点? 内存会不会爆?**

不测就动手, 最常见的后果是优化了不是瓶颈的那一环 ——
比如花两天做解析并行化, 结果发现 90% 的时间在 embedding 上.

用法::

    python backend/scripts/bench_large_pdf.py --pages 200
    python backend/scripts/bench_large_pdf.py --pages 500 --skip-embed   # 只测解析
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from contextlib import suppress
from pathlib import Path

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        with suppress(Exception):
            _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

#: 合成文档每一页的正文模板. 内容长度刻意接近真实技术文档的一页.
_PAGE_TEMPLATE = """第 {page} 章 设备维护规范

{section}

本设备在生产过程中需要经过清洗站、投板站、AOI 站等多个工站。每个工站都有
明确的作业标准与检验要求，操作人员必须严格按照规范执行并如实记录。设备维护
的核心目标是保证产品良率稳定，同时把非计划停机时间控制在可接受范围内。

{body}

维护记录的保存期限为三年，期间任何修改都必须留下痕迹。质量部门每季度会对
维护记录进行一次抽查，抽查比例不低于百分之十。对于抽查中发现的问题，责任
部门需要在五个工作日内提交整改方案并落实。
"""

_BODY_TEMPLATE = (
    "钢刀的更换周期为 {n} 次或 {m} 个月，以先到者为准。更换时必须核对型号，"
    "并在维护记录中登记更换时间、操作人员与旧件编号。锡膏的储存温度必须控制在"
    "2 到 10 摄氏度之间，使用前需要回温至少三十分钟。清洗站的校准周期为每周一次，"
    "由当班工程师负责执行并签字确认。"
)


#: 段落素材. 反复取用直到填满一页 —— 真实技术文档每页约 1400 字.
#:
#: ⚠️ **每条都必须带占位符**. 第一版里有 7 条是固定文本, 结果它们在每页完全相同,
#: 被清洗阶段的"跨页重复内容去重"当成模板噪音删掉了 —— 2800 个块只剩 505 个,
#: 82% 的内容凭空消失, 压测数字完全失真.
#:
#: 这反过来验证了去重逻辑是生效的, 也暴露了一个真实风险:
#: **长文档里跨页重复的正常内容(例如每章都重复的操作须知)有被误删的可能**.
#: 详见 docs/07-大文档处理方案.md.
_PARAGRAPHS = (
    "本设备在生产过程中需要经过清洗站、投板站、AOI 站等多个工站，第 {n} 号工站的作业标准与检验要求如下所述。",
    "钢刀的更换周期为 {n} 次或 {m} 个月，以先到者为准，更换时必须核对型号并登记操作人员与旧件编号。",
    "锡膏的储存温度必须严格控制在 {m} 到 10 摄氏度之间，第 {n} 批物料使用前需要回温至少三十分钟。",
    "清洗站的校准周期为每 {m} 周一次，由当班工程师负责执行并签字确认，校准记录保存三年备查。",
    "维护记录的任何修改都必须留下痕迹，记录编号 {n} 的修改需注明修改人、修改时间与修改原因。",
    "对于抽查中发现的问题，责任部门需要在 {m} 个工作日内提交整改方案并落实，逾期纳入绩效考核。",
    "设备非计划停机时间按月统计，第 {n} 号机台超过目标值时需提交原因分析与改善措施并复查。",
    "新员工上岗前必须完成第 {n} 阶段岗前培训与考核，考核不合格者不得独立操作设备。",
)


def _page_lines(page_no: int, count: int = 38) -> list[str]:
    """生成一页的文本行. 每行都带页号, 保证跨页不重复(见 _PARAGRAPHS 的注释)."""
    lines: list[str] = []
    for i in range(count):
        template = _PARAGRAPHS[i % len(_PARAGRAPHS)]
        lines.append(template.format(n=20000 + page_no * 10 + i, m=3 + (page_no + i) % 6))
    return lines


def build_pdf(path: Path, pages: int, *, realistic: bool = True) -> int:
    """生成一份结构规整的合成 PDF, 返回字节数.

    **realistic=True 是必须的**. 第一版用 ``insert_textbox`` + 内置西文字体生成,
    结果 200 页只要 0.17 秒(1 ms/页) —— 而真实的 2 页简历解析要 1.6 秒(790 ms/页),
    差了 790 倍. 原因有三个:

    1. 内置字体不嵌入, PyMuPDF 直接映射到标准字体; 真实 PDF 常用**子集嵌入字体**
       (尤其 Type3 逐字嵌入), 每个字形都是一个 PDF 对象, 解析成本高一个数量级
    2. 一个 ``insert_textbox`` 只产生**一个文本块**; 真实文档每页有几十个块
       (标题、正文、页眉、页脚、列表项各算一块), 定位与排序开销随之上升
    3. 每页字符数差 6 倍(220 vs 1438)

    所以这里改成: 嵌入 CJK 字体 + 每页几十个独立文本块 + 每页约 1450 字.

    ⚠️ 即便如此, 这个合成文档仍然是**理想情况**: 没有表格、没有图片、
       没有 Type3 逐字嵌入字体. 测量结果只能作为**下界**参考.
    """
    import pymupdf as fitz

    doc = fitz.open()
    for page_no in range(1, pages + 1):
        page = doc.new_page(width=595, height=842)

        # 页眉(跨页重复, 用来验证页眉页脚剔除在长文档上的行为)
        page.insert_text((56, 40), "设备维护规范 内部资料", fontsize=8, fontname="china-s")
        page.insert_text(
            (56, 78), f"第 {page_no} 章  设备维护规范", fontsize=16, fontname="china-s"
        )
        page.insert_text(
            (56, 100), f"{page_no}.1  钢刀与锡膏的更换标准", fontsize=12, fontname="china-s"
        )

        if not realistic:
            page.insert_textbox(
                fitz.Rect(56, 120, 539, 786),
                "\n".join(_page_lines(page_no)),
                fontsize=10,
            )
            page.insert_text((280, 812), f"第 {page_no} 页", fontsize=8, fontname="china-s")
            continue

        # 正文: 每行一个独立文本块, 填满整页
        y = 122.0
        for line in _page_lines(page_no, count=38):
            if y > 790:
                break
            page.insert_text((56, y), line, fontsize=9.5, fontname="china-s")
            y += 17.5

        # 页脚(跨页重复)
        page.insert_text(
            (270, 812), f"第 {page_no} 页 共 {pages} 页", fontsize=8, fontname="china-s"
        )

    doc.save(str(path), deflate=True)
    doc.close()
    return path.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description="大文档吞吐压测")
    parser.add_argument("--pages", type=int, default=200, help="合成的页数")
    parser.add_argument("--pdf", default=None, help="用已有的 PDF 而不是合成")
    parser.add_argument("--skip-embed", action="store_true", help="跳过向量化(只测解析与分块)")
    parser.add_argument("--keep", action="store_true", help="保留下合成的 PDF")
    args = parser.parse_args()

    from app.core.config import settings
    from app.services.chunking import ChunkParams, chunk_document
    from app.services.parser import clean_document, parse_pdf

    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"[FAIL] 文件不存在: {pdf_path}")
            return 1
    else:
        pdf_path = PROJECT_ROOT / "bench" / f"bench_{args.pages}p.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"生成 {args.pages} 页合成 PDF…")
        size = build_pdf(pdf_path, args.pages)
        print(f"  文件大小: {size / 1024 / 1024:.2f} MB\n")

    def measure(label: str, fn):
        gc.collect()
        started = time.perf_counter()
        result = fn()
        cost = time.perf_counter() - started
        print(f"  {label:<16} {cost:8.3f} s")
        return result, cost

    print("=" * 64)
    print(" 阶段耗时")
    print("=" * 64)

    parsed, parse_s = measure("① 解析 PDF", lambda: parse_pdf(pdf_path))
    print(
        f"     └─ {parsed.page_count} 页, {parsed.char_count:,} 字符, "
        f"每页 {parse_s / max(parsed.page_count, 1) * 1000:.0f} ms"
    )

    cleaned, clean_s = measure("② 清洗", lambda: clean_document(parsed))
    print(f"     └─ {len(cleaned.paragraphs):,} 段落")

    params = ChunkParams.from_settings()
    chunking, chunk_s = measure("③ 分块", lambda: chunk_document(cleaned, "BENCH", params=params))
    print(f"     └─ {len(chunking.parents):,} 父块 / {len(chunking.children):,} 子块")

    embed_s = 0.0
    if not args.skip_embed:
        from app.services.embedding import get_embedding_provider

        provider = get_embedding_provider()
        texts = [c.content for c in chunking.children]

        # 预热: 首次推理包含模型加载与显存分配, 必须排除掉,
        # 否则测出来的是"冷启动"而不是"吞吐"
        provider.encode_passages(texts[: min(8, len(texts))])

        batch_size = settings.embedding_batch_size
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        def run_embed() -> float:
            started = time.perf_counter()
            for batch in batches:
                provider.encode_passages(batch)
            return time.perf_counter() - started

        _, embed_s = measure(f"④ 向量化(批={batch_size})", run_embed)
        if texts:
            print(
                f"     └─ {len(texts):,} 子块, {embed_s / len(texts) * 1000:.2f} ms/块, "
                f"{len(texts) / max(embed_s, 1e-6):,.0f} 块/秒"
            )

    total = parse_s + clean_s + chunk_s + embed_s
    print("\n" + "=" * 64)
    print(" 汇总")
    print("=" * 64)
    print(f"  总耗时          : {total:.2f} s")
    print(f"  解析占比        : {parse_s / total * 100:.1f}%")
    print(f"  清洗占比        : {clean_s / total * 100:.1f}%")
    print(f"  分块占比        : {chunk_s / total * 100:.1f}%")
    if embed_s:
        print(f"  向量化占比      : {embed_s / total * 100:.1f}%")

    if parsed.page_count:
        per_page = total / parsed.page_count
        print(f"\n  单页均摊        : {per_page:.3f} s/页")
        print("  外推:")
        for target in (50, 100, 200, 500, 1000):
            seconds = per_page * target
            flag = "  ← 已超过网关 60s 超时" if seconds > 60 else ""
            print(f"    {target:>5} 页  ≈ {seconds:7.1f} s ({seconds / 60:5.1f} 分钟){flag}")

    print(f"\n  当前嵌入模型    : {settings.embedding_model} device={settings.embedding_device}")
    print(
        f"  当前分块参数    : child={settings.child_chunk_size} parent={settings.parent_chunk_size}"
    )

    if not args.keep and not args.pdf:
        with suppress(OSError):
            pdf_path.unlink()
        print("  (已删除临时 PDF; 用 --keep 保留)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
