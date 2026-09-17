"""朗读口语化转换的单元测试.

这一层的测试价值特别高, 因为**它的正确性没法靠"看"验证**:
代码写得再对, 也得实际听一遍才知道读出来是什么效果.
测试把"应该读成什么"固化下来, 就不需要每次改都戴上耳机听.

用例分成四类:
1. Markdown 清理   —— 星号井号这些不该读出来
2. 引用编号清理    —— RAG 的 [1] 对听众是噪声
3. 技术缩写        —— 少数几个读错率高的
4. **不该动的**    —— 这条最容易被忽略: 好心的过度转换会引入新 bug
"""

from __future__ import annotations

import pytest

from app.services.speech.normalizer import to_spoken


# --------------------------------------------------------------------------- #
# 1. Markdown 清理
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("**为什么**要用 Redis？", "为什么要用 Redis？"),
        ("这是*重点*内容", "这是重点内容"),
        ("***又粗又斜***的文本", "又粗又斜的文本"),
        ("~~删掉的话~~不算", "删掉的话不算"),
        ("__下划线加粗__", "下划线加粗"),
    ],
)
def test_strips_inline_markdown(raw, expected):
    """星号/下划线必须清掉 —— 否则 TTS 会读成"星号星号为什么星号星号"。"""
    assert to_spoken(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("# 标题\n正文", "标题\n正文"),
        ("## 二级标题", "二级标题"),
        ("- 第一点\n- 第二点", "第一点\n第二点"),
        ("* 星号列表", "星号列表"),
        ("1. 有序一\n2. 有序二", "有序一\n有序二"),
        ("> 引用内容", "引用内容"),
    ],
)
def test_strips_line_prefixes(raw, expected):
    assert to_spoken(raw) == expected


def test_removes_fenced_code_block():
    """代码块读出来没有任何意义, 整段丢弃 —— 而不是读出代码内容。"""
    raw = "看这段代码：\n```python\nprint('hello')\n```\n就这些。"
    out = to_spoken(raw)
    assert "print" not in out
    assert "hello" not in out
    assert "看这段代码" in out and "就这些" in out


def test_keeps_inline_code_content_without_backticks():
    """行内代码要**保留内容**只去掉反引号 —— 它通常是个技术名词, 该读出来。"""
    assert to_spoken("用了 `Chroma` 做向量库") == "用了 Chroma 做向量库"


def test_link_keeps_text_drops_url():
    assert to_spoken("见[官方文档](https://example.com/a/b)") == "见官方文档"


def test_bare_url_is_removed():
    out = to_spoken("参考 https://example.com/very/long/path 里面的说明")
    assert "http" not in out
    assert "参考" in out and "里面的说明" in out


def test_removes_emoji():
    assert to_spoken("完成 ✅ 了 🎉") == "完成 了"
    assert "📌" not in to_spoken("📌 注意这里")


def test_table_separator_removed():
    out = to_spoken("| 参数 | 值 |\n|---|---|\n| 温度 | 200 |")
    assert "---" not in out


# --------------------------------------------------------------------------- #
# 2. 引用编号
#
# RAG 链路的 [1][2] 对**看**答案的人是来源标记, 对**听**答案的人只是噪声 ——
# 听众看不到引用卡片, 读出"中括号一"毫无意义.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("换刀周期是 20000 次[1]。", "换刀周期是 20000 次。"),
        ("温度要求 2~10 度[1][2]。", "温度要求 2到10 度。"),
        ("见前文[1,2]。", "见前文。"),
        ("参考[1-3]这几处。", "参考这几处。"),
        ("写成 [ 1 ] 也要能识别。", "写成 也要能识别。"),
    ],
)
def test_removes_citation_markers(raw, expected):
    assert to_spoken(raw) == expected


def test_does_not_remove_non_citation_brackets():
    """只有纯数字的方括号才是引用编号。

    「[文字]」这类是正常内容, 不能一起删掉 —— 正则写宽一点就会误伤。
    """
    out = to_spoken("这里[重点]要记住")
    assert "重点" in out


# --------------------------------------------------------------------------- #
# 3. 技术缩写
# --------------------------------------------------------------------------- #
def test_expands_qps_to_letters():
    """QPS 连写时部分音色会读成"库普斯", 拆成字母更稳。"""
    assert "Q P S" in to_spoken("你的 QPS 是多少？")


def test_tech_term_replace_is_word_bounded():
    """带边界的替换 —— 否则 QPS 会命中 XQPSS 这种意外子串。"""
    out = to_spoken("变量名是 myQPSvalue")
    assert "Q P S" not in out


def test_expands_common_latin_abbreviations():
    out = to_spoken("用了 Redis, e.g. 缓存, vs. 数据库")
    assert "例如" in out
    assert "对比" in out


# --------------------------------------------------------------------------- #
# 4. 不该动的东西（最重要的一类）
#
# 好心的过度转换会引入新 bug. 现代 TTS 本来就能正确读数字、百分比、
# 英文单词, 手工转换只会把对的改错.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "响应时间提升了 40%",
        "P95 从 800ms 降到 120ms",
        "写入了 3000 条记录",
        "2023 年 9 月到 2024 年 3 月",
        "MySQL 和 Redis 都是 8.0 版本",
        "用了 Spring Boot 3.2.1",
        "第 1 题的第 2 小问",
    ],
)
def test_leaves_numbers_and_units_alone(text):
    """数字、百分比、单位、技术名词**原样保留**.

    现代神经 TTS 读这些本来就是对的(40% → "百分之四十"),
    手工转成中文数字反而会把"1/2"这类多义写法改错.
    """
    assert to_spoken(text) == text


def test_keeps_chinese_punctuation():
    raw = "这是第一句。这是第二句！还有第三句？"
    assert to_spoken(raw) == raw


# --------------------------------------------------------------------------- #
# 5. 边界情况
# --------------------------------------------------------------------------- #
def test_empty_input_returns_empty():
    assert to_spoken("") == ""
    assert to_spoken("   \n  ") == ""


def test_code_only_input_returns_empty():
    """全是代码块 → 清完什么都不剩.

    调用方必须据此**跳过合成**, 而不是播一段静音 ——
    用户会以为程序卡住了.
    """
    assert to_spoken("```python\nprint(1)\n```") == ""


def test_truncates_at_sentence_boundary():
    """超长时在句末截断, 而不是硬切。

    硬切会把"为什么"切成"为什", 读出来是个残缺的词。
    """
    text = "第一句话在这里。第二句话在这里。第三句话在这里。"
    out = to_spoken(text, max_chars=14)
    assert out.endswith("。")
    assert "第二句话" not in out


def test_truncate_falls_back_to_hard_cut_when_no_boundary():
    """没有句末标点时只能硬切 —— 但不能因为找不到标点就返回空串。"""
    out = to_spoken("一二三四五六七八九十" * 5, max_chars=10)
    assert 0 < len(out) <= 10


def test_collapses_whitespace_and_stray_brackets():
    """删掉内容后留下的空括号与多余空白要收拾干净, 否则会有莫名停顿。"""
    out = to_spoken("温度是（）度")
    assert "（）" not in out
    assert "()" not in out


def test_multiline_becomes_readable():
    """多行文本要保留换行 —— 换行在 TTS 里是一次自然停顿, 比逗号更合适。"""
    out = to_spoken("- 第一点\n- 第二点\n- 第三点")
    assert out.count("\n") == 2


def test_numeric_range_reads_as_dao():
    """数字之间的波浪号是**区间**, 必须读成"到"。

    这是"删掉标记"类规则里最容易出事的一种: 标记本身承载了语义。
    简单把 ~ 删掉会得到「210 度」—— 意思完全变了。
    """
    assert to_spoken("温度 2~10 度") == "温度 2到10 度"
    assert to_spoken("2～10 度") == "2到10 度"


def test_arrow_between_terms_becomes_a_pause():
    """「A → B」读成停顿（逗号）, 而不是"到"。

    这里试过读成"到", 但在**真实文本**上立刻出问题:
    「提升了 40% → 这个数字怎么测的」读成「40%到这个数字怎么测的」很别扭。

    箭头的语义本来分两种 —— 「A 到 B」(范围/流程)和「A, 然后 B」(转折),
    只看字符判断不了是哪种. 而逗号在两种语境下都自然。
    """
    assert to_spoken("版面分析 → OCR") == "版面分析，OCR"
    assert to_spoken("提升了 40% → 怎么测的") == "提升了 40%，怎么测的"


def test_arrow_after_sentence_end_is_dropped():
    """句末的箭头连逗号都不该补 —— 那里已经有句号了。

    「你做过吗？→ 说说…」补成「你做过吗？，说说…」会多一个莫名其妙的停顿。
    """
    out = to_spoken("你做过这个吗？→ 说说具体怎么做的")
    assert "？，" not in out
    assert "说说具体怎么做的" in out
