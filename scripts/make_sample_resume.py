"""生成示例简历 PDF，用于演示「模拟面试」功能。

两种模式
--------
**默认（手工版）** —— 一份精心设计的简历，每个"槽点"都在源码里标了出来:

- 有量化数字但没说来源(「提升 40%」怎么算的?)
- 有技术栈但没说选型理由(为什么用 Chroma 不用 FAISS?)
- 有前后不一致(实习写"负责后端", 项目又写"独立完成全栈")
- 有模糊表述(「优化了性能」「参与了」)

这些**才是面试官该追问的地方**。拿一份说明书去面试, 面试官只能问
"这个参数是多少", 完全没有意义 —— 文档本身就是给追问准备的靶子.

**随机模式 (`--random`)** —— 每次生成不同的人、学校、公司、项目和技术栈,
用于反复演示或造多份文档。

关键设计: **随机化的是内容, 保留的是"可被追问的结构"**。
一份通顺但无懈可击的简历demo 起来是没戏的 —— 面试官找不到下手的地方,
只能问"介绍一下你的项目"。所以随机生成器仍然会注入那几类槽点,
并在结束时**告诉你这份简历埋了哪些点**, 方便演示时心里有数。

敏感信息
--------
演示材料绝不能带真实个人信息。这里从三个层面保证:

1. **池子里全是虚构内容** —— 姓名取常见称呼, 学校/公司一律用「某某…」
2. **联系方式是显式占位符** —— `138-0000-0000` / `@example.com`
   (`example.com` 是 RFC 2606 保留域名, 永远不可能指向真人)
3. **生成后做一次自检** —— 扫一遍产物, 发现任何像真实手机号/邮箱/身份证的
   模式就直接报错。**工具要能证明自己的产出是安全的**, 而不是"我觉得应该没事".

用法::

    python scripts/make_sample_resume.py                       # 手工版
    python scripts/make_sample_resume.py --random              # 随机一份
    python scripts/make_sample_resume.py --random --seed 42    # 可复现
    python scripts/make_sample_resume.py --random --count 3    # 一次生成 3 份
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = REPO_ROOT / "samples" / "张明-后端开发-示例简历.pdf"

PAGE_W, PAGE_H = 595, 842  # A4
MARGIN = 56


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class ResumeContent:
    """一份简历的全部内容. 排版与内容分开, 两种模式才能共用同一套 PDF 生成代码."""

    name: str
    contact: str
    sections: list[tuple[str, list[str]]]
    #: 这份简历埋了哪些"可被追问的点". 打印给演示者看 ——
    #: 知道靶子在哪, demo 时才能顺势引导面试官 Agent 去追.
    probes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 手工版（默认）
#
# 刻意保留的"槽点"都用注释标出来了 —— 这不是随便写的文本, 是给面试官出题用的。
# 改内容时请保持这些槽点, 否则面试会立刻退化成"流水账复述"。
# --------------------------------------------------------------------------- #
CRAFTED = ResumeContent(
    name="张明",
    contact="手机 138-0000-0000 ｜ 邮箱 zhangming@example.com ｜ 求职意向：后端开发工程师",
    sections=[
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
    ],
    probes=[
        "「响应时间提升 40%」没给基线与测量方法",
        "「负责订单系统后端开发」范围模糊",
        "实习写「负责后端」而项目写「独立完成全栈」，口径不一致",
        "个人项目却给出 MRR 0.988，规模感不匹配",
        "自我评价全是大词，零具体证据",
    ],
)


# --------------------------------------------------------------------------- #
# 随机版
#
# 全部为虚构内容. 学校与公司一律用「某某」, 联系方式是 RFC 2606 保留域名
# 与全零号码 —— 目的是让任何看到的人**一眼就知道不是真人**.
# --------------------------------------------------------------------------- #
#: 姓名与邮箱前缀**配对固定**.
#:
#: 不能各自随机 —— 会出现"赵鑫"配"zhangming@example.com"这种对不上的组合,
#: 一眼就显得是随便编的. 演示材料的可信度靠细节, 这种低级不一致会让人
#: 怀疑整份简历都是假的(虽然它确实是), 从而不相信后面的追问是"读懂了简历".
NAMES: tuple[tuple[str, str], ...] = (
    ("张明", "zhangming"),
    ("李婷", "liting"),
    ("王浩", "wanghao"),
    ("陈雨", "chenyu"),
    ("刘洋", "liuyang"),
    ("赵鑫", "zhaoxin"),
    ("孙悦", "sunyue"),
    ("周航", "zhouhang"),
    ("吴迪", "wudi"),
    ("徐蕾", "xulei"),
)
SCHOOLS = ("某某大学", "某某理工大学", "某某工业大学", "某某科技大学", "某某师范大学")
MAJORS = ("计算机科学与技术", "软件工程", "人工智能", "数据科学与大数据技术", "电子信息工程")
COMPANIES = (
    "某某科技有限公司",
    "某某网络科技有限公司",
    "某某信息技术有限公司",
    "某某数据服务有限公司",
)

#: 起止时间池 —— 只造出"看起来合理的年份", 不涉及任何真实经历
RANGES = ("2021.07 - 2021.12", "2022.03 - 2022.08", "2022.07 - 2022.12", "2022.09 - 2023.02")
PROJECT_RANGES = ("2023.03 - 2023.08", "2023.05 - 2023.10", "2023.09 - 2024.02")

#: 每条 {role} 会被替换成求职方向. 括号里的词标出了这是哪一类"追问靶子".
TRACKS: dict[str, dict[str, list[str]]] = {
    "后端开发": {
        "skills": [
            "编程语言：Java、Python、Go（了解）",
            "Web 框架：Spring Boot、FastAPI",
            "数据库：MySQL、Redis、PostgreSQL",
            "中间件：RabbitMQ、Kafka（了解）",
            "其他：Docker、Git、Linux 常用命令",
        ],
        "intern_bullets": [
            # 靶子: 模糊职责
            "· 负责{city}业务系统的后端开发，参与需求评审与接口设计",
            # 靶子: 无口径的量化
            "· 通过引入 Redis 缓存热点数据，将{metric}提升了 {pct_up}%",
            # 靶子: 只有动作没有结果
            "· 优化了慢 SQL，对{表}添加联合索引，解决了线上偶发的超时问题",
            "· 使用 RabbitMQ 处理{场景}超时取消，替代原有的定时任务轮询方案",
        ],
        "project_bullets": [
            "· 独立完成全栈开发，实现{链路}的完整链路",
            "· 使用 FastAPI 搭建后端，接口平均响应 {ms} ms",
            # 靶子: 选型没说理由
            "· 向量库选用 Chroma（嵌入式部署），嵌入模型使用 BGE-small-zh",
        ],
        #: 指标类的一句话单独放 —— **必须和项目类型对得上**.
        #: 之前是从项目描述里按关键词挑, 结果给"数据看板系统"配了
        #: "检索 MRR 达到 0.995" —— 一个看板哪来的检索指标?
        "metric_bullets": [
            "· 自建压测脚本，接口 QPS 达到 {qps}",
            "· 优化后 P95 从 {ms_big} ms 降到 {ms_small} ms",
        ],
    },
    "AI 应用开发": {
        "skills": [
            "AI 应用：RAG、Prompt 工程、Function Calling、Agent 工作流",
            "编程语言：Python、Java（了解）",
            "框架：LangChain、FastAPI、PyTorch（了解）",
            "向量检索：Chroma、Milvus、BM25",
            "其他：Docker、Git、Linux 常用命令",
        ],
        "intern_bullets": [
            "· 负责{city}知识库问答系统的落地，参与方案设计与效果评估",
            # 靶子: 无口径的量化
            "· 通过优化分块策略，将检索命中率提升至 {pct_level}%",
            "· 整理样本评测集，对模型回答正确率做量化分析并输出评估报告",
            "· 沉淀 Prompt 模板与{场景}处理流程，降低人工答疑成本",
        ],
        "project_bullets": [
            "· 设计「版面分析 → OCR → 语义分块 → 向量化检索」链路",
            # 靶子: 不说取舍
            "· 使用 BGE-small-zh 做本地嵌入，数据不出域",
            "· 检索链路：向量召回 + BM25 关键词召回，用 RRF 融合后重排",
        ],
        #: RAG 类项目才用检索指标 —— 见后端方向里关于"指标要配对"的说明
        "metric_bullets": [
            "· 自建评测集 {n} 条，检索 MRR 达到 0.{mrr}",
            "· 文档问答检索命中率提升至 {pct_level}%",
        ],
    },
    "前端开发": {
        "skills": [
            "编程语言：TypeScript、JavaScript、Python（了解）",
            "框架：Vue 3、React、Nuxt",
            "工程化：Vite、Webpack、ESLint、pnpm",
            "样式：TailwindCSS、Sass",
            "其他：Git、Docker、Linux 常用命令",
        ],
        "intern_bullets": [
            "· 负责{city}管理后台的前端开发，参与组件库建设",
            # 靶子: 无口径的量化
            "· 通过虚拟列表与懒加载，将首屏渲染时间降低了 {pct_up}%",
            "· 优化了打包配置，把产物体积从 {big} MB 压到 {small} MB",
            "· 使用 WebSocket 实现{场景}实时推送，替代原有的轮询方案",
        ],
        "project_bullets": [
            "· 独立完成一个{Role}类应用，从需求拆解到上线全流程",
            "· 实现了{链路}，支持 {n} 条数据的流畅滚动",
            "· 使用 Pinia 管理状态，抽象了统一的请求层与错误处理",
        ],
        #: 前端用性能评分, 不用检索指标
        "metric_bullets": [
            "· 首屏性能评分达到 {score} 分（Lighthouse）",
            "· 打包产物体积从 {big} MB 压到 {small} MB",
        ],
    },
    "数据开发": {
        "skills": [
            "编程语言：Python、SQL、Scala（了解）",
            "大数据：Spark、Flink、Hive",
            "存储：MySQL、ClickHouse、Redis",
            "调度：Airflow、DolphinScheduler",
            "其他：Docker、Git、Linux 常用命令",
        ],
        "intern_bullets": [
            "· 负责{city}数据仓库的{表}层建设，参与指标口径评审",
            # 靶子: 无口径的量化
            "· 通过拆分大宽表与预聚合，将核心报表产出时间缩短了 {pct_up}%",
            "· 优化了慢查询，对{表}表调整分区与索引",
            "· 使用 Airflow 编排{场景}离线任务，替代原有的 shell 定时脚本",
        ],
        "project_bullets": [
            "· 独立完成{链路}的链路搭建，覆盖采集、清洗、入库",
            "· 使用 Spark 处理日均 {n} 万条数据",
            "· 设计了分区与索引策略，查询响应稳定在 {ms} ms 以内",
        ],
        #: 数据方向用数据质量指标
        "metric_bullets": [
            "· 自建数据质量校验规则 {n} 条，异常拦截率达到 {pct_level}%",
            "· 核心报表产出时间从 {min_big} 分钟缩短到 {min_small} 分钟",
        ],
    },
}

#: 「优化前后」的数字必须**成对抽取**.
#:
#: 第一版把「大」和「小」各自随机, 结果出现了
#: 「P95 从 20 ms 降到 200 ms」和「产出时间从 20 分钟缩短到 200 分钟」——
#: 优化完反而更慢了. 这种错不是"有槽点", 是**纯粹的低级错误**,
#: 会让人觉得整份简历是乱编的, 反而破坏演示可信度.
#:
#: 成对之后, 大小关系在构造上就不可能出现问题, 不依赖"记得调池子".
_PAIRS_MS = ((600, 120), (800, 200), (1200, 300), (1500, 260))
_PAIRS_MIN = ((30, 5), (45, 8), (60, 12), (90, 15))

#: 每条都会被塞进简历的"模糊措辞" —— 面试官看到这些就该追问
SELF_EVAL = [
    "学习能力强，对新技术有较强的钻研意愿，能够快速上手陌生技术栈。",
    "具备良好的团队协作能力，在项目中承担了主要的开发工作。",
    "沟通表达清晰，能够主动推进问题解决，具备较强的自驱力。",
    "对{role}方向有持续的热情，习惯用工具提升开发效率。",
]

FILLERS = {
    "city": ("电商", "订单", "支付", "会员", "物流", "风控"),
    "metric": ("订单查询接口响应时间", "核心接口 P95 耗时", "首页加载时间"),
    "表": ("订单", "交易", "用户", "商品"),
    "场景": ("订单", "任务", "消息", "审批"),
    "链路": ("文档上传、向量化、检索问答", "用户下单、支付、履约", "数据采集、清洗、入库"),
    "Role": ("知识库问答", "任务管理", "数据看板"),
}


def _fill(template: str, rng: random.Random) -> str:
    """把模板里的占位符替换成随机取值, 并随机化其中的数字."""
    out = template
    for key, options in FILLERS.items():
        out = out.replace("{" + key + "}", rng.choice(options))
    out = out.replace("{pct}", str(rng.choice((30, 35, 40, 45, 50, 60, 70))))
    # 百分比要按**语义位置**分池 —— 共用一个池子会造出
    # "Lighthouse 评分 30 分"(30 分是差评)、"命中率提升至 50%"(提升后还只有一半)
    # 这类荒谬数字. 它们不是"有槽点", 是纯粹的低级错误, 反而毁掉演示可信度.
    #
    #   pct_up    改进幅度: "提升了 / 降低了 N%"
    #   pct_level 绝对水平: "达到 / 提升至 N%"
    #   score     评分(0~100, 且低分是坏消息)
    out = out.replace("{pct_up}", str(rng.choice((30, 35, 40, 45, 50, 60, 70))))
    out = out.replace("{pct_level}", str(rng.choice((88, 91, 93, 95, 96))))
    out = out.replace("{score}", str(rng.choice((86, 89, 92, 95, 98))))
    out = out.replace("{ms}", str(rng.choice((80, 120, 150, 200, 260))))
    out = out.replace("{n}", str(rng.choice((30, 46, 60, 80, 100, 120))))
    out = out.replace("{mrr}", f"{rng.randint(930, 995)}")
    out = out.replace("{big}", str(rng.choice((8, 12, 15))))
    out = out.replace("{small}", str(rng.choice((3, 4, 5, 6))))
    out = out.replace("{qps}", str(rng.choice((800, 1200, 1500, 2000, 3000))))
    # 成对替换: 大小关系由 _PAIRS_* 保证, 不靠"记得把池子调对"
    big_ms, small_ms = rng.choice(_PAIRS_MS)
    out = out.replace("{ms_big}", str(big_ms)).replace("{ms_small}", str(small_ms))
    big_min, small_min = rng.choice(_PAIRS_MIN)
    return out.replace("{min_big}", str(big_min)).replace("{min_small}", str(small_min))


def random_resume(
    rng: random.Random, role: str | None = None, name: tuple[str, str] | None = None
) -> ResumeContent:
    """随机拼一份简历.

    **随机化内容, 但不随机化"结构"** —— 每份都会带上那几类槽点, 因为它们
    才是面试官 Agent 的用武之地. 一份无懈可击的简历 demo 起来是没戏的.

    Args:
        rng: 随机源(定种子可复现)
        role: 指定方向; None 则随机
        name: 指定 (姓名, 邮箱前缀); None 则随机。
            一次生成多份时由调用方轮流传入, 保证不重样。
    """
    role = role or rng.choice(list(TRACKS))
    track = TRACKS[role]
    name, email_prefix = name or rng.choice(NAMES)

    edu_years = rng.choice(("2019.09 - 2023.06", "2020.09 - 2024.06"))
    gpa = f"{rng.randint(30, 39) / 10:.1f}"
    rank = rng.choice((10, 15, 20, 25, 30))
    school = rng.choice(SCHOOLS)
    major = rng.choice(MAJORS)

    intern_company = rng.choice(COMPANIES)
    intern_range = rng.choice(RANGES)
    project_range = rng.choice(PROJECT_RANGES)

    # 实习: 抽 3 条, 保证至少带一条"无口径的量化"
    intern_pool = [_fill(b, rng) for b in track["intern_bullets"]]
    quantified = [b for b in intern_pool if "%" in b]
    rest = [b for b in intern_pool if "%" not in b]
    rng.shuffle(rest)
    intern_bullets = ([quantified[0]] if quantified else []) + rest[:2]

    project_pool = [_fill(b, rng) for b in track["project_bullets"]]
    rng.shuffle(project_pool)
    # 指标类句子从这个方向**专属**的池子里抽, 保证和项目类型对得上
    project_bullets = project_pool[:2] + [_fill(rng.choice(track["metric_bullets"]), rng)]

    sections: list[tuple[str, list[str]]] = [
        (
            "教育背景",
            [
                f"{edu_years}  {school}  {major}  本科",
                "主修课程：数据结构、操作系统、计算机网络、数据库系统原理",
                f"绩点 {gpa}/4.0，专业排名前 {rank}%",
            ],
        ),
        ("专业技能", list(track["skills"])),
        (
            "实习经历",
            [
                f"{intern_range}  {intern_company}  {role}实习生",
                *intern_bullets,
            ],
        ),
        (
            "项目经历",
            [
                f"{project_range}  {_fill('{Role}系统', rng)}（个人项目）",
                *project_bullets,
                "",
                f"{rng.choice(('2022.03 - 2022.06', '2022.09 - 2023.01'))}  "
                f"校园{_fill('{场景}', rng)}平台（课程项目，{rng.randint(2, 4)} 人小组）",
                f"· 负责后端接口开发与数据库表设计，使用 {rng.choice(('Spring Boot + MySQL', 'FastAPI + PostgreSQL', 'Node.js + MongoDB'))}",
                "· 实现了信息发布、状态流转、用户评价等核心功能",
                rng.choice(("· 项目获得课程优秀项目奖", "· 课程设计评分 92 分")),
            ],
        ),
        ("自我评价", [_fill(t.replace("{role}", role), rng) for t in rng.sample(SELF_EVAL, 3)]),
    ]

    probes = [
        f"「{role}」方向下的量化数字都没有基线与测量方法",
        "实习职责写得像岗位说明书（「负责…，参与…」），看不出本人做了什么",
        "个人项目给出接近满分的指标，规模感不匹配",
        "技术选型只说了用了什么，没说为什么不用别的",
        "自我评价全是大词，零具体证据",
    ]
    return ResumeContent(
        name=name,
        # 联系方式是**显式占位符**: 全零号码 + RFC 2606 保留域名。
        # 用 example.com 而不是随便编一个域名, 是因为它被标准保留,
        # 永远不会指向真人邮箱 —— 演示材料不该有一丝可能误伤真实的人.
        contact=(
            f"手机 138-0000-0000 ｜ 邮箱 {email_prefix}@example.com ｜ 求职意向：{role}工程师"
        ),
        sections=sections,
        probes=probes,
    )


# --------------------------------------------------------------------------- #
# 敏感信息自检
# --------------------------------------------------------------------------- #
#: 像真实手机号的模式: 1 开头 11 位, 且**不是**全零占位符
_RE_PHONE = re.compile(r"1[3-9]\d{9}")
#: 18 位身份证
_RE_ID_CARD = re.compile(r"\b\d{17}[\dXx]\b")
#: 任何邮箱
_RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
#: 允许的邮箱域名(RFC 2606 保留域名, 永远不会指向真人)
_SAFE_EMAIL_DOMAINS = ("example.com", "example.org", "example.net", "test.invalid")


def check_no_pii(text: str) -> list[str]:
    """扫一遍产物, 返回发现的可疑项.

    演示材料里的个人信息是最容易出事的地方 —— 而且往往是"从某个模板抄来的"
    这种无意识的泄漏。所以不靠"我写的时候注意了", 而是**生成后机器验一遍**。
    """
    problems: list[str] = []

    for match in _RE_PHONE.finditer(text):
        problems.append(f"像真实手机号: {match.group(0)}")
    for match in _RE_ID_CARD.finditer(text):
        problems.append(f"像身份证号: {match.group(0)}")
    for match in _RE_EMAIL.finditer(text):
        domain = match.group(0).rsplit("@", 1)[-1].lower()
        if domain not in _SAFE_EMAIL_DOMAINS:
            problems.append(f"非保留域名的邮箱: {match.group(0)}")
    return problems


#: 模板里未替换的占位符, 形如 {metric}
_RE_PLACEHOLDER = re.compile(r"\{[a-zA-Z_]+\}")


def check_placeholders(text: str) -> list[str]:
    """检查有没有没填上的占位符.

    这个检查很必要: 模板里用了 ``{foo}`` 而 ``_fill`` 忘了替换的话,
    **``{foo}`` 会原样印进 PDF**。它不报错、生成也"成功",
    只有人眼逐行看才发现 —— 而这种东西出现在演示材料里相当尴尬.
    """
    return [f"未替换的占位符: {m}" for m in _RE_PLACEHOLDER.findall(text)]


def resume_text(content: ResumeContent) -> str:
    """把简历拼成纯文本, 用于自检与预览."""
    lines = [content.name, content.contact, ""]
    for title, items in content.sections:
        lines.append(title)
        lines.extend(items)
        lines.append("")
    return "\n".join(lines)


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


def build_resume(content: ResumeContent, path: Path) -> int:
    """生成简历 PDF, 返回页数."""
    import pymupdf as fitz

    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y = MARGIN

    # ---------------- 头部: 姓名 + 联系方式 ----------------
    page.insert_text((MARGIN, y), content.name, fontsize=19, fontname="china-s")
    y += 26
    page.insert_text((MARGIN, y), content.contact, fontsize=9, fontname="china-s")
    y += 14
    page.draw_line((MARGIN, y), (PAGE_W - MARGIN, y), width=0.8)
    y += 24

    def new_page():
        return doc.new_page(width=PAGE_W, height=PAGE_H)

    for title, lines in content.sections:
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


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def _safe_filename(content: ResumeContent, role_hint: str = "") -> str:
    """用姓名 + 求职方向做文件名. 姓名本身是虚构的, 不构成隐私问题."""
    role = role_hint
    if not role:
        # 从联系方式里取"求职意向"
        match = re.search(r"求职意向：(.+?)(?:工程师)?$", content.contact)
        role = match.group(1).strip() if match else "示例"
    return f"{content.name}-{role}-示例简历.pdf"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="生成示例简历 PDF（演示用，全部为虚构内容）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("out", nargs="?", type=Path, default=None, help="输出路径(单份时用)")
    parser.add_argument("--random", action="store_true", help="随机生成(默认用手工设计的那份)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子, 用于复现同一份")
    parser.add_argument("--count", type=int, default=1, help="生成几份(--random 时可用)")
    parser.add_argument("--role", default=None, help=f"指定方向: {' / '.join(TRACKS)}")
    parser.add_argument("--quiet", action="store_true", help="不打印追问点")
    args = parser.parse_args()

    try:
        import pymupdf  # noqa: F401
    except ImportError:
        print("[X] 缺少 PyMuPDF, 请先运行 install.bat / install.sh")
        return 1

    if args.count < 1:
        print("[X] --count 至少为 1")
        return 1
    if args.role and args.role not in TRACKS:
        print(f"[X] 未知方向 {args.role}, 可选: {', '.join(TRACKS)}")
        return 1
    if args.out and args.count > 1:
        print("[X] --count > 1 时不要指定输出路径(会自动按姓名命名)")
        return 1
    if args.count > 1 and not args.random:
        print("[X] --count > 1 需要配合 --random, 否则每份内容都一样")
        return 1

    # 没给种子时用一个"可复现的随机" —— 把实际种子打出来, 用户想再要一份同样的
    # 就能照着输入. 纯随机会让"刚才那份挺好的, 再来一份"变成不可能.
    seed = args.seed if args.seed is not None else random.randrange(1, 10**6)
    rng = random.Random(seed)

    # 一次生成多份时**主动保证多样性**, 而不是指望随机抽到不同方向 ——
    # 实测 seed=2026 的前四次恰好全是"后端开发", 概率虽小但真实发生,
    # 而拿到三份一模一样的简历对演示毫无意义. 多份的场景本来就是要多样性.
    role_cycle: list[str] = []
    name_cycle: list[tuple[str, str]] = []
    if args.count > 1:
        role_cycle = list(TRACKS)
        rng.shuffle(role_cycle)
        name_cycle = list(NAMES)
        rng.shuffle(name_cycle)

    out_dir = REPO_ROOT / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    made: list[tuple[Path, int, ResumeContent]] = []
    for index in range(args.count):
        if args.random:
            # 多份时按打乱后的列表轮着取, 保证方向和姓名都不重样
            forced_role = args.role or (role_cycle[index % len(role_cycle)] if role_cycle else None)
            content = random_resume(
                rng,
                role=forced_role,
                name=(name_cycle[index % len(name_cycle)] if name_cycle else None),
            )
        else:
            content = CRAFTED

        if args.out:
            target = args.out
        elif args.count == 1:
            target = DEFAULT_TARGET if not args.random else out_dir / _safe_filename(content)
        else:
            target = out_dir / _safe_filename(content)

        # 同名就加序号 —— 随机模式撞名(同名同方向)是可能的
        if target.exists() and args.random and target not in {p for p, _, _ in made}:
            target = target.with_name(f"{target.stem}-{index + 1}{target.suffix}")

        target.parent.mkdir(parents=True, exist_ok=True)

        # ---------------- 生成前先自检 ----------------
        text = resume_text(content)
        problems = check_no_pii(text) + check_placeholders(text)
        if problems:
            print(f"[X] 第 {index + 1} 份内容自检没通过, 已中止:")
            for item in problems:
                print(f"      - {item}")
            return 1

        pages = build_resume(content, target)
        made.append((target, pages, content))

    print(f"[OK] 已生成 {len(made)} 份示例简历" + (f"（种子 {seed}）" if args.random else ""))
    print()
    for target, pages, _ in made:
        print(f"  {target.relative_to(REPO_ROOT)}")
        print(f"    {pages} 页 / {target.stat().st_size / 1024:.0f} KB")

    if not args.quiet:
        # 把靶子打出来: demo 时知道该往哪引导, 而不是等面试官自己撞上去
        print("\n这份简历埋了这些可被追问的点（演示时心里有数）:")
        for probe in made[0][2].probes:
            print(f"  · {probe}")

    print()
    print("已做敏感信息自检: 手机号/身份证/非保留域名邮箱 均未发现")
    print()
    print("下一步:")
    print("  1. 启动服务 (start.bat)")
    print("  2. 「文档管理」上传这份 PDF, 等状态变成「已就绪」")
    print("  3. 切到「模拟面试」, 选这份简历 + 勾选 SKILL, 开始")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
