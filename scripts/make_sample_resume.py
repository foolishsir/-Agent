"""生成一份示例简历 PDF, 用于体验「模拟面试」功能.

为什么单独做一个简历样例
------------------------
项目自带的 ``NX-3000设备维护手册`` 是用来演示**问答**的 —— 它是一份说明书,
结构规整、事实密集, 适合验证检索与引用.

但**面试**要的是另一种文档: 有模糊表述、有可质疑的数字、有前后不一致的经历 ——
这些才是面试官该追问的地方. 拿说明书去面试, 面试官只能问"这个参数是多少",
完全没有意义.

所以这里专门造一份"有槽点"的简历:

- 有量化数字但没说来源(「提升 40%」怎么算的?)
- 有技术栈但没说选型理由(为什么用 Chroma 不用 FAISS?)
- 有前后不一致(实习写"负责后端", 项目又写"独立完成全栈")
- 有模糊表述(「优化了性能」「参与了」)

有了这些槽点, 面试官 Agent 的追问逻辑才有东西可追 ——
文档本身就是给追问准备的**靶子**.

用法::

    python scripts/make_sample_resume.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "samples" / "张明-后端开发-示例简历.pdf"

PAGE_W, PAGE_H = 595, 842  # A4
MARGIN = 56

# --------------------------------------------------------------------------- #
# 简历内容
#
# 刻意保留的"槽点"都用注释标出来了 —— 这不是随便写的文本, 是给面试官
# 出题用的. 改内容时请保持这些槽点, 否则面试会立刻退化成"流水账复述".
# --------------------------------------------------------------------------- #
NAME = "张明"
CONTACT = "手机 138-0000-0000 ｜ 邮箱 zhangming@example.com ｜ 求职意向：后端开发工程师"

SECTIONS: list[tuple[str, list[str]]] = [
    (
        "教育背景",
        [
            "2019.09 - 2023.06  某某大学  计算机科学与技术  本科",
            "主修课程：数据结构、操作系统、计算机网络、数据库系统原理",
            "绩点 3.6/4.0，专业排名前 20%",
        ],
    ),
    (
        "专业技能",
        [
            "编程语言：Python、Java、Go（了解）",
            "Web 框架：FastAPI、Spring Boot",
            "数据库：MySQL、Redis、PostgreSQL",
            "中间件：RabbitMQ、Kafka（了解）",
            "其他：Docker、Git、Linux 常用命令",
        ],
    ),
    (
        "实习经历",
        [
            "2022.07 - 2022.12  某某科技有限公司  后端开发实习生",
            # 槽点: 「负责」很模糊 —— 到底写了多少? 独立做的还是打下手?
            "· 负责订单系统的后端开发，参与需求评审与接口设计",
            # 槽点: 40% 没说口径 —— 什么指标? 什么基线? 怎么测的?
            "· 通过引入 Redis 缓存热点数据，将订单查询接口响应时间提升了 40%",
            # 槽点: 「优化了」是模糊词, 没说优化了什么
            "· 优化了慢 SQL，对订单表添加联合索引，解决了线上偶发的超时问题",
            "· 使用 RabbitMQ 处理订单超时取消，替代原有的定时任务轮询方案",
        ],
    ),
    (
        "项目经历",
        [
            "2023.03 - 2023.09  个人知识库问答系统（独立开发）",
            # 槽点: 「独立完成全栈」与实习的「负责后端」形成对比 —— 会写前端吗?
            "· 独立完成全栈开发，实现文档上传、向量化、检索问答的完整链路",
            "· 使用 PyMuPDF 解析 PDF，设计了父子分块策略提升检索精度",
            "· 向量库选用 Chroma（嵌入式部署），嵌入模型使用 BGE-small-zh",
            "· 检索链路：向量召回 + BM25 关键词召回，用 RRF 融合后送入 CrossEncoder 重排",
            # 槽点: 这个数字与"个人项目"的规模感不匹配, 值得确认
            "· 自建评测集 46 条，检索 MRR 达到 0.988",
            "",
            "2022.03 - 2022.06  校园二手交易平台（课程项目，3 人小组）",
            "· 负责后端接口开发与数据库表设计，使用 Spring Boot + MySQL",
            "· 实现了商品发布、订单流转、用户评价等核心功能",
            "· 项目获得课程优秀项目奖",
        ],
    ),
    (
        "自我评价",
        [
            # 槽点: 全是大词, 没有一个具体证据 —— 面试官一定会追问
            "学习能力强，对新技术有较强的钻研意愿，能够快速上手陌生技术栈。",
            "具备良好的团队协作能力，在课程项目中承担了主要的开发工作。",
            "沟通表达清晰，能够主动推进问题解决。",
        ],
    ),
]


# --------------------------------------------------------------------------- #
# 排版
# --------------------------------------------------------------------------- #
def _wrap(text: str, width: int = 44) -> list[str]:
    """按字符数折行. 中英文混排按"中文 1 格、英文 0.6 格"粗估宽度."""
    if not text:
        return [""]
    lines, cur, w = [], "", 0.0
    for ch in text:
        cw = 1.0 if ord(ch) > 0x2E80 else 0.6
        if w + cw > width:
            lines.append(cur)
            cur, w = ch, cw
        else:
            cur += ch
            w += cw
    if cur:
        lines.append(cur)
    return lines


def build_resume(path: Path) -> int:
    """生成简历 PDF, 返回页数."""
    import pymupdf as fitz

    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y = MARGIN

    # ---------------- 头部: 姓名 + 联系方式 ----------------
    page.insert_text((MARGIN, y), NAME, fontsize=19, fontname="china-s")
    y += 26
    page.insert_text((MARGIN, y), CONTACT, fontsize=9, fontname="china-s")
    y += 14
    page.draw_line((MARGIN, y), (PAGE_W - MARGIN, y), width=0.8)
    y += 24

    def new_page():
        return doc.new_page(width=PAGE_W, height=PAGE_H)

    for title, lines in SECTIONS:
        # 章节标题前留白; 空间不够就翻页(标题不孤行)
        if y > PAGE_H - 140:
            page = new_page()
            y = MARGIN

        page.insert_text((MARGIN, y), title, fontsize=12.5, fontname="china-s")
        y += 6
        page.draw_line((MARGIN, y), (MARGIN + 44, y), width=1.4)
        y += 16

        for line in lines:
            if not line:
                y += 8
                continue
            for wrapped in _wrap(line):
                if y > PAGE_H - MARGIN - 20:
                    page = new_page()
                    y = MARGIN
                # 正文用 9.5pt: 比标题(12.5pt)明显小, 保证标题检测能靠"字号 + 单行"识别出来
                page.insert_text((MARGIN, y), wrapped, fontsize=9.5, fontname="china-s")
                y += 15
            y += 3
        y += 12

    doc.save(str(path))
    count = doc.page_count
    doc.close()
    return count


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else TARGET
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pymupdf  # noqa: F401
    except ImportError:
        print("[X] 缺少 PyMuPDF, 请先运行 install.bat / install.sh")
        return 1

    pages = build_resume(target)
    size_kb = target.stat().st_size / 1024
    print(f"[OK] 已生成示例简历: {target}")
    print(f"     {pages} 页 / {size_kb:.0f} KB")
    print()
    print("下一步:")
    print("  1. 启动服务 (start.bat)")
    print("  2. 「文档管理」上传这份 PDF, 等状态变成「已就绪」")
    print("  3. 切到「模拟面试」, 选这份简历 + 勾选 SKILL, 开始")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
