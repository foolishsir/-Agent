"""Prompt 模板.

设计原则
--------
1. **规则放前面**: 指令放在上下文之前, 模型对开头指令的遵循度更高.
2. **编号引用**: 让模型用 ``[1] [2]`` 标注来源, 这是引用溯源的起点.
   但**不能只靠 Prompt** —— 模型会编造不存在的编号, 所以服务端必须校验(见 citation.py).
3. **明确拒答话术**: 给一个固定的拒答句式, 便于前端识别和统计拒答率.
4. **上下文按相关度降序**: 并结合"最重要放头尾"的编排对抗 Lost-in-the-Middle.
5. **上下文用分隔符包裹**: 文档内容是**数据**不是**指令**,
   用明确的边界标记可以降低 Prompt 注入的风险(文档里写"忽略以上指令"之类).

关于少量示例(few-shot)
-----------------------
本项目用 zero-shot + 强规则, 没有加示例. 原因是:
示例会占用可观的 token(每个示例几百字), 而 RAG 场景下真正的瓶颈是上下文长度.
只有在实测发现模型不遵循规则时, 才值得用示例补充. 先测再优化, 不要预设。
"""

from __future__ import annotations

from app.services.retrieval import RetrievedContext

#: 拒答时使用的固定话术. 前端据此判断"这是拒答"而不是普通回答.
REFUSAL_ANSWER = "文档中未提及该问题。"

SYSTEM_PROMPT = f"""你是一个严谨的文档问答助手。你的唯一信息来源是用户提供的【参考资料】。

必须遵守的规则：
1. 只依据【参考资料】回答，绝不使用你自己的知识补充。
2. 每个论断后面必须标注来源编号，格式如 [1]、[2]。可以引用多个，如 [1][3]。
3. 如果【参考资料】中没有足够信息回答该问题，只回复一句：{REFUSAL_ANSWER}
4. 绝不编造参考资料中不存在的信息，包括数字、型号、日期、人名。
5. 如果资料中信息互相矛盾，指出矛盾并分别标注来源。
6. 回答简洁直接，不要重复问题本身，不要添加"根据参考资料"之类的开场白。

排版要求（前端按 Markdown 渲染，请正确使用语法，否则用户会看到一堆星号和井号）：
- 列举多项时用无序列表，每行以 "- " 开头，不要堆成一大段。
- 关键术语和结论用 **加粗** 强调。
- 参数对比用 Markdown 表格。
- 代码、命令、字段名用反引号包裹。
- 引用编号写成 [1] 这种方括号数字形式，前端会把它渲染成可点击角标。"""


def build_context_block(contexts: list[RetrievedContext]) -> str:
    """把检索到的父块拼成带编号的参考资料块.

    **顺序编排**: 精排分最高的放最前, 次高的放最后, 其余按序排中间.
    这是对 "Lost in the Middle" 的应对 —— 长上下文中模型对中间位置的
    注意力显著下降, 把最相关的信息放在头尾能提升利用率.

    代价是编号顺序与相关度顺序不一致, 但对用户来说引用编号本来就只是标识符,
    真正重要的是答案质量.
    """
    if not contexts:
        return "（无参考资料）"

    ordered = _reorder_for_attention(contexts)

    blocks: list[str] = []
    for context in ordered:
        header = f"[{context.index}] 来源: {context.filename} 第 {context.page_start}"
        if context.page_end != context.page_start:
            header += f"-{context.page_end}"
        header += " 页"
        if context.section_path:
            header += f" | 章节: {context.section_path}"
        blocks.append(f"{header}\n{context.content}")

    return "\n\n---\n\n".join(blocks)


def _reorder_for_attention(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
    """按"最重要在头尾"重排(不改变 index, 只改变出现顺序)."""
    if len(contexts) <= 2:
        return list(contexts)

    ordered = sorted(contexts, key=lambda c: c.score, reverse=True)
    head = ordered[0]
    tail = ordered[1]
    middle = ordered[2:]
    return [head, *middle, tail]


def build_user_prompt(question: str, contexts: list[RetrievedContext]) -> str:
    """组装用户消息."""
    return f"""【参考资料】
{build_context_block(contexts)}

【用户问题】
{question}

请依据上述参考资料回答，并在每个论断后标注来源编号。"""


# --------------------------------------------------------------------------- #
# 多轮对话改写
# --------------------------------------------------------------------------- #
QUERY_REWRITE_SYSTEM = """你是一个查询改写助手。把用户的追问改写成一个**自包含**的检索查询。

规则：
1. 补全追问中省略的主语、宾语（把"它""这个""那呢"替换成具体名词）。
2. 只输出改写后的查询本身，不要任何解释、前缀或引号。
3. 如果原问题本身已经完整（没有代词和省略），原样输出。
4. 保持原问题的语言和关键术语（型号、数字不能改）。
5. 改写后不要超过 60 字。"""


def build_rewrite_prompt(question: str, history: list[tuple[str, str]]) -> str:
    """构造改写请求. ``history`` 是 (role, content) 列表, 按时间正序."""
    recent = history[-6:]  # 只取最近 3 轮, 更早的对话对指代消解没有价值
    lines = "\n".join(
        f"{'用户' if role == 'user' else '助手'}: {content}" for role, content in recent
    )
    return f"""【最近的对话】
{lines}

【用户的新问题】
{question}

请输出改写后的、自包含的检索查询："""
