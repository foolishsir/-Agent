"""PDF 解析器 —— 基于 PyMuPDF 的版式感知解析.

核心问题
--------
PDF 只描述"某个字符画在哪个坐标", **不描述阅读顺序**.
``page.get_text()`` 按 PDF 内部对象顺序输出, 不保证是人类阅读顺序.
双栏论文会左右栏交错读出乱序, 页眉页脚会混进正文.

本模块的处理链路
----------------
1. 用 ``get_text("dict")`` 提取**文本块 + 坐标 + 字号**
2. **跨页重复检测**自动识别页眉页脚(不写死"去掉前 50 像素")
3. **投影法检测分栏**, 双栏页面按栏还原阅读顺序
4. **字号 + 编号正则**识别标题, 为后续父子块切分提供结构
5. 抽取字符数过少 → 判定扫描件并明确上报

每一步都是"不依赖具体文档模板"的通用启发式, 换一份 PDF 不需要重新调参.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

from app.core.exceptions import DocumentParseError
from app.core.logging import get_logger, log_kv
from app.services.parser.base import ParsedDocument, ParsedPage, TextBlock

try:  # PyMuPDF >= 1.24 推荐 ``import pymupdf``, 旧版本只有 ``fitz``
    import pymupdf as fitz
except ImportError:  # pragma: no cover - 取决于安装的 PyMuPDF 版本
    import fitz  # type: ignore[no-redef]

logger = get_logger("docmind.parser.pdf")

# --------------------------------------------------------------------------- #
# 标题识别规则
# --------------------------------------------------------------------------- #
# 注意: 不能用 ``^\d+\s`` 这种宽松规则 —— "1333 机种" 这类以数字开头的正文
# 会被误判成标题. 所以要求编号必须**带点分级**(如 3.2 / 3.2.1)或使用中文序号.
_SECTION_NUMBER_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,3}){1,3})\s*\S")
_CHINESE_NUMBER_RE = re.compile(r"^\s*(第\s*[一二三四五六七八九十百零〇\d]{1,6}\s*[章节条部分篇])")
_CHINESE_LIST_RE = re.compile(r"^\s*([一二三四五六七八九十]{1,3})\s*[、.．]")
_BRACKET_NUMBER_RE = re.compile(r"^\s*[（(]\s*[一二三四五六七八九十\d]{1,3}\s*[)）]")

#: 标题的长度上限 —— 超过这个长度即使字号大也更可能是正文强调句
_HEADING_MAX_CHARS = 60

#: 判定标题需要比正文大出的**绝对**字号差(磅).
#: 为什么用绝对差而不是纯比率: 真实文档里标题往往只比正文大 0.5~1 磅
#: (实测某份简历正文 9.4pt / 标题 10.4pt, 比率仅 1.11, 用 1.15 倍阈值会全部漏掉);
#: 而纯比率在正文字号很小时又会变得过于敏感. 两者取较大值更稳.
_HEADING_MIN_DELTA_PT = 0.8

#: 正文与标题的字号差至少要达到正文字号的这个比例(大字号文档用)
_HEADING_DELTA_RATIO = 0.06

#: 投影法检测分栏时的分箱数
_COLUMN_BINS = 40


def parse_pdf(path: str | Path, *, filename: str | None = None) -> ParsedDocument:
    """解析 PDF 文件.

    Args:
        path: PDF 文件路径
        filename: 展示用的文件名(默认取路径的文件名)

    Raises:
        DocumentParseError: 文件无法打开或解析失败
    """
    file_path = Path(path)
    display_name = filename or file_path.name

    try:
        raw = file_path.read_bytes()
    except OSError as exc:
        raise DocumentParseError(f"读取文件失败: {exc}") from exc

    try:
        doc = fitz.open(stream=raw, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - PyMuPDF 抛的异常类型不稳定
        raise DocumentParseError(f"PDF 打开失败, 文件可能已损坏: {exc}") from exc

    try:
        if doc.needs_pass:
            raise DocumentParseError("PDF 已加密, 请先解除密码保护后再上传")

        pages = [_extract_page(page, index + 1) for index, page in enumerate(doc)]
        metadata = _extract_metadata(doc)
    finally:
        doc.close()

    # ---- 步骤 1: 跨页重复行检测, 识别页眉页脚 ----
    running_texts = _detect_running_headers(pages)
    if running_texts:
        log_kv(
            logger,
            "parser.headers_detected",
            count=len(running_texts),
            pages=len(pages),
            sample=list(running_texts)[:3],
        )
        for page in pages:
            page.blocks = [
                b for b in page.blocks if _normalize_for_repeat(b.text) not in running_texts
            ]

    # ---- 步骤 2: 分栏检测 + 阅读顺序还原 ----
    for page in pages:
        page.blocks = _order_blocks(page)

    # ---- 步骤 3: 标题识别 ----
    body_font_size = _body_font_size(pages)
    for page in pages:
        page.blocks = [_mark_heading(b, body_font_size) for b in page.blocks]

    # 丢掉清洗后为空的块
    for page in pages:
        page.blocks = [b for b in page.blocks if b.text.strip()]

    result = ParsedDocument(
        filename=display_name,
        pages=pages,
        metadata=metadata,
    )

    # ---- 步骤 4: 扫描件判定 ----
    # 不静默返回空结果 —— 用户上传完看到"就绪"却问不出东西是最差的体验.
    pages_with_text = sum(1 for p in pages if p.text_length >= 20)
    if pages and pages_with_text / len(pages) < 0.5:
        result.is_scanned = True
        log_kv(
            logger,
            "parser.scanned_detected",
            pages=len(pages),
            pages_with_text=pages_with_text,
        )

    log_kv(
        logger,
        "parser.done",
        file=display_name,
        pages=len(pages),
        chars=result.char_count,
        scanned=result.is_scanned,
    )
    return result


# --------------------------------------------------------------------------- #
# 页面抽取
# --------------------------------------------------------------------------- #
def _extract_page(page: fitz.Page, page_no: int) -> ParsedPage:
    """把一页转成 ParsedPage(文本块 + 坐标 + 字号)."""
    parsed = ParsedPage(
        page_no=page_no,
        width=float(page.rect.width),
        height=float(page.rect.height),
    )

    try:
        raw = page.get_text("dict")
    except Exception as exc:  # noqa: BLE001 - 单页失败不应中断整份文档
        logger.warning("第 %s 页文本抽取失败, 已跳过 | error=%s", page_no, exc)
        return parsed

    for block in raw.get("blocks", []):
        # type: 0 = 文本块, 1 = 图片块. 图片内容暂不处理(见文档中的"已知边界")
        if block.get("type") != 0:
            continue

        lines: list[str] = []
        sizes: list[float] = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans)
            if line_text.strip():
                lines.append(line_text)
            sizes.extend(
                float(span.get("size", 0.0)) for span in spans if span.get("text", "").strip()
            )

        text = "\n".join(lines).strip()
        if not text:
            continue

        bbox = block.get("bbox") or (0.0, 0.0, 0.0, 0.0)
        parsed.blocks.append(
            TextBlock(
                text=text,
                page_no=page_no,
                x0=float(bbox[0]),
                y0=float(bbox[1]),
                x1=float(bbox[2]),
                y1=float(bbox[3]),
                # 一个块里可能有多种字号, 取最大值代表它 —— 标题块通常整体字号偏大
                font_size=round(max(sizes), 2) if sizes else 0.0,
            )
        )

    return parsed


def _extract_metadata(doc: fitz.Document) -> dict[str, object]:
    """抽取 PDF 元信息(标题/作者等), 仅用于展示."""
    try:
        meta = doc.metadata or {}
    except Exception:  # noqa: BLE001
        return {}
    return {k: v for k, v in meta.items() if v}


# --------------------------------------------------------------------------- #
# 页眉页脚: 跨页重复行检测
# --------------------------------------------------------------------------- #
def _normalize_for_repeat(text: str) -> str:
    """归一化文本用于重复检测.

    把数字统一替换成 ``#``, 这样 "第 1 页 / 共 12 页" 和 "第 7 页 / 共 12 页"
    会归一化成同一个 key, 从而被识别为页脚.

    这是本方法能"通用"的关键 —— 不需要知道页码格式是什么样.
    """
    return re.sub(r"\d+", "#", text.strip()).lower()


def _detect_running_headers(
    pages: list[ParsedPage],
    *,
    top_n: int = 3,
    bottom_n: int = 3,
    min_page_ratio: float = 0.5,
) -> set[str]:
    """找出在多页顶部/底部重复出现的文本.

    为什么用重复检测而不是固定位置裁剪:
    不同文档的页眉高度、页脚边距都不一样, 写死位置的话每换一份文档就要重新调参.
    而"重复出现"是页眉页脚的**定义性特征**, 与具体排版无关.

    保守起见:
    - 文档少于 3 页时不做检测(样本太少, 误杀风险高)
    - 要求覆盖率 >= 50% 的页面
    """
    if len(pages) < 3:
        return set()

    counter: Counter[str] = Counter()
    for page in pages:
        if not page.blocks:
            continue
        ordered = sorted(page.blocks, key=lambda b: b.y0)
        candidates = ordered[:top_n] + ordered[-bottom_n:]
        # 同一页内去重: 同一行在一页里出现两次只算一次
        for normalized in {_normalize_for_repeat(b.text) for b in candidates if b.text.strip()}:
            counter[normalized] += 1

    threshold = max(2, int(len(pages) * min_page_ratio))
    # 只保留长度合理的候选 —— 太长的"重复文本"更可能是正文而非常规页眉
    return {text for text, count in counter.items() if count >= threshold and 0 < len(text) <= 80}


# --------------------------------------------------------------------------- #
# 分栏检测与阅读顺序还原
# --------------------------------------------------------------------------- #
def _order_blocks(page: ParsedPage) -> list[TextBlock]:
    """按人类阅读顺序重排页面内的文本块.

    单栏: 按 y 再按 x 排序.
    双栏: 先读左栏(自上而下), 再读右栏.

    注意: 这是**启发式**而非精确还原. PDF 里并没有"栏"这个概念,
    我们只是根据文本块的 x 分布做推断, 对绝大多数规整排版的文档有效.
    """
    blocks = page.blocks
    if len(blocks) < 6:  # 块太少时统计不可靠, 不做分栏猜测
        return sorted(blocks, key=lambda b: (round(b.y0, 1), b.x0))

    gutter = _detect_gutter(blocks, page.width)
    if gutter is None:
        return sorted(blocks, key=lambda b: (round(b.y0, 1), b.x0))

    # 跨栏块: 横跨分栏缝隙, 且宽度明显超过半页(典型的跨栏标题 / 宽表格 / 跨栏图).
    # 用 id() 而不是值做集合: TextBlock 是 frozen dataclass, 内容相同的两块
    # 会被判为相等, 用值去重会误删正常内容.
    spanning = [b for b in blocks if b.x0 < gutter < b.x1 and b.width > page.width * 0.55]
    spanning_ids = {id(b) for b in spanning}
    left = [b for b in blocks if id(b) not in spanning_ids and b.x_center < gutter]
    right = [b for b in blocks if id(b) not in spanning_ids and b.x_center >= gutter]

    if not left or not right:
        return sorted(blocks, key=lambda b: (round(b.y0, 1), b.x0))

    sort_key = lambda b: (round(b.y0, 1), b.x0)  # noqa: E731 - 局部小工具, 定义成函数反而绕
    left.sort(key=sort_key)
    right.sort(key=sort_key)

    # 跨栏块按纵向区间归属到某一栏, 避免被错误地提到页面最前面.
    # 例: 位于页面中部的跨栏表格, 应该插在左栏内容之后, 而不是排到最开头.
    left_top, left_bottom = left[0].y0, max(b.y1 for b in left)
    right_top, right_bottom = right[0].y0, max(b.y1 for b in right)

    for block in sorted(spanning, key=sort_key):
        y = block.y_center
        covers_left = left_top <= y <= left_bottom
        covers_right = right_top <= y <= right_bottom
        if covers_left and not covers_right:
            left.append(block)
        elif covers_right and not covers_left:
            right.append(block)
        else:
            # 两栏区间都覆盖(如页面顶部的跨栏标题) → 归到起始位置更靠前的那一栏
            (left if left_top <= right_top else right).append(block)

    left.sort(key=sort_key)
    right.sort(key=sort_key)
    return left + right


def _detect_gutter(blocks: list[TextBlock], page_width: float) -> float | None:
    """用一维投影法找分栏缝隙的 x 坐标; 找不到(即单栏)返回 None.

    做法: 把页宽分成若干等分, 统计每个等分被文本块覆盖的次数.
    双栏排版会在页面中间形成一条**零覆盖的竖直空白带**(栏间距 gutter).
    """
    if page_width <= 0:
        return None

    coverage = [0] * _COLUMN_BINS
    for block in blocks:
        i0 = int(block.x0 / page_width * _COLUMN_BINS)
        i1 = int(block.x1 / page_width * _COLUMN_BINS)
        for i in range(max(0, i0), min(_COLUMN_BINS - 1, i1) + 1):
            coverage[i] += 1

    # 只在页面中部 30%~70% 的范围内找 —— 页边距也会形成空白带, 但不在中间
    lo, hi = int(_COLUMN_BINS * 0.30), int(_COLUMN_BINS * 0.70)
    best_run = best_start = 0
    run = 0
    for i in range(lo, hi):
        if coverage[i] == 0:
            run += 1
            if run > best_run:
                best_run, best_start = run, i - run + 1
        else:
            run = 0

    # 缝隙至少要有 2 个分箱宽(约页宽的 5%), 否则可能只是词间空格造成的噪声
    if best_run < 2:
        return None

    gutter = (best_start + best_run / 2) / _COLUMN_BINS * page_width

    # 两侧都必须有实质内容, 否则只是"左边有字右边空着"
    left_count = sum(1 for b in blocks if b.x_center < gutter)
    right_count = len(blocks) - left_count
    if left_count < 2 or right_count < 2:
        return None

    # 还要检查是否真有"只属于某一栏"的块 —— 如果所有块都是跨栏的宽块, 那是单栏
    narrow_left = sum(1 for b in blocks if b.x_center < gutter and b.x1 <= gutter * 1.05)
    narrow_right = sum(1 for b in blocks if b.x_center >= gutter and b.x0 >= gutter * 0.95)
    if narrow_left < 2 or narrow_right < 2:
        return None

    return gutter


# --------------------------------------------------------------------------- #
# 标题识别
# --------------------------------------------------------------------------- #
def _body_font_size(pages: list[ParsedPage]) -> float:
    """估计正文字号.

    用**按字符数加权的众数**而不是中位数或均值:

    - 均值会被少量大字号标题拉高, 导致真正的标题反而"不够大"
    - 中位数在小文档上不稳定, 且在双峰分布(正文 + 大量小字号注释)时会落在两峰之间

    众数天然锁定"出现最多的那个字号", 而正文在字符数上必然占绝对多数.
    单块权重上限 300 字符, 避免一个超长段落独自决定结果.
    """
    weights: Counter[float] = Counter()
    for page in pages:
        for block in page.blocks:
            if block.font_size > 0:
                weights[round(block.font_size, 1)] += min(len(block.text), 300)

    if not weights:
        return 0.0
    return weights.most_common(1)[0][0]


def _mark_heading(block: TextBlock, body_font_size: float) -> TextBlock:
    """判断文本块是否为标题, 返回带标记的新块."""
    text = block.text.strip()
    if not text or len(text) > _HEADING_MAX_CHARS:
        return block

    # 规则 0(硬性): 标题必须是**单行**.
    # 这一条排除了 PDF 里最常见的误判来源 —— 表格/表单的标签块.
    # 例如简历里 "年　　龄\n22 岁\n性　　别\n男" 会被 PyMuPDF 合并成一个块,
    # 字号偏大但不是标题. 单行限制把它们全部挡掉.
    if "\n" in text:
        return block

    # 规则 1: 字号明显大于正文
    delta = max(_HEADING_MIN_DELTA_PT, body_font_size * _HEADING_DELTA_RATIO)
    font_heading = body_font_size > 0 and block.font_size >= body_font_size + delta

    # 规则 2: 匹配编号样式(章节号 / 中文序号 / 括号序号).
    # 即使字号与正文相同也能识别 —— 结构化文档普遍用编号标记层级.
    numbered = bool(
        _SECTION_NUMBER_RE.match(text)
        or _CHINESE_NUMBER_RE.match(text)
        or _CHINESE_LIST_RE.match(text)
        or _BRACKET_NUMBER_RE.match(text)
    )

    if not (font_heading or numbered):
        return block

    # 以句末标点结尾的更可能是正文的一句话
    if text.endswith(("。", "！", "？", ".", ";", "；")):
        return block

    return replace(block, is_heading=True)


def extract_section_number(text: str) -> str | None:
    """从标题文本中提取编号部分(如 "3.2"), 用于拼接章节路径."""
    for pattern in (_SECTION_NUMBER_RE, _CHINESE_LIST_RE, _BRACKET_NUMBER_RE):
        match = pattern.match(text.strip())
        if match:
            return match.group(1).strip()
    match = _CHINESE_NUMBER_RE.match(text.strip())
    if match:
        return re.sub(r"\s+", "", match.group(1))
    return None
