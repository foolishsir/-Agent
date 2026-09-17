"""示例简历生成器的测试.

重点是**"演示材料里不能有敏感信息"**这条硬要求 —— 它不能靠"我写的时候注意了",
必须机器验。所以这里既验生成结果干净, 也**反向验证检查器本身有效**
(一个从不报警的检查等于没有检查)。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

# scripts/ 不是包, 但生成器就在那儿 —— 为测试单独造一个包结构不值当,
# 直接把目录挂上 sys.path. 加注释是因为这种写法在别处看到会让人困惑.
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from make_sample_resume import (  # noqa: E402
    CRAFTED,
    NAMES,
    TRACKS,
    check_no_pii,
    check_placeholders,
    random_resume,
    resume_text,
)


# --------------------------------------------------------------------------- #
# 敏感信息
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "should_flag"),
    [
        ("手机 138-0000-0000 ｜ 邮箱 zhangming@example.com", False),
        ("邮箱 a@example.org", False),
        ("联系电话 13812345678", True),  # 真实手机号
        ("身份证 110101199003071234", True),
        ("邮箱 zhangsan@qq.com", True),  # 非保留域名
        ("邮箱 someone@gmail.com", True),
    ],
)
def test_pii_checker_catches_real_looking_pii(text, should_flag):
    """**反向验证检查器**: 它必须真的会报警, 否则这条防线是假的.

    只测"干净内容通过"是不够的 —— 一个永远返回空的函数也能通过那种测试.
    """
    assert bool(check_no_pii(text)) is should_flag


def test_crafted_resume_has_no_pii():
    """手工版也要过自检 —— 它是默认生成的那份."""
    assert check_no_pii(resume_text(CRAFTED)) == []


@pytest.mark.parametrize("role", list(TRACKS))
@pytest.mark.parametrize("seed", [1, 7, 42, 99, 2026])
def test_random_resume_is_always_clean(role, seed):
    """每个方向 x 多个种子都必须干净.

    随机生成的东西最容易漏 —— 池子里混进一条带真实手机号的样例,
    平时抽不到, 抽到了就是事故. 所以覆盖全部方向 + 多个种子.
    """
    text = resume_text(random_resume(random.Random(seed), role=role))
    assert check_no_pii(text) == []


@pytest.mark.parametrize("seed", range(20))
def test_generated_text_has_no_unfilled_placeholders(seed):
    """模板占位符必须全部被替换.

    漏填的 ``{foo}`` 会**原样印进 PDF**, 不报错、生成也"成功",
    只有人眼逐行看才发现. 出现在演示材料里相当尴尬.
    """
    text = resume_text(random_resume(random.Random(seed)))
    assert check_placeholders(text) == []


def test_contact_uses_reserved_domain_only():
    """联系方式只能用 RFC 2606 保留域名.

    example.com 被标准保留, **永远不可能指向真人邮箱** ——
    演示材料不该有一丝可能误伤真实的人. 随便编一个域名就有这个风险.
    """
    for seed in range(20):
        contact = random_resume(random.Random(seed)).contact
        assert "@example.com" in contact
        assert "138-0000-0000" in contact


# --------------------------------------------------------------------------- #
# 内容一致性
#
# 这类问题不报错, 但会让演示材料一眼看着"是随便编的" ——
# 反而破坏可信度, 让人觉得面试官的追问不是"读懂了简历".
# --------------------------------------------------------------------------- #
def test_email_prefix_matches_the_name():
    """姓名和邮箱前缀必须配对.

    各自随机会出现"赵鑫"配"zhangming@example.com"这种对不上的组合.
    """
    mapping = dict(NAMES)
    for seed in range(20):
        content = random_resume(random.Random(seed))
        prefix = content.contact.split("邮箱 ", 1)[1].split("@", 1)[0]
        assert mapping[content.name] == prefix, f"{content.name} 配了 {prefix}"


def test_improvement_numbers_actually_improve():
    """「从 A 降到 B」必须 A > B.

    第一版把大小两个数字各自随机, 造出了「P95 从 20 ms 降到 200 ms」——
    优化完反而更慢了. 这不是"有槽点", 是纯粹的低级错误.
    """
    import re

    pattern = re.compile(r"从 (\d+) (?:ms|分钟) (?:降|缩短)到 (\d+) ")
    checked = 0
    for role in TRACKS:
        for seed in range(40):
            text = resume_text(random_resume(random.Random(seed), role=role))
            for before, after in pattern.findall(text):
                checked += 1
                assert int(before) > int(after), f"{role}/{seed}: 从 {before} 到 {after} 没有变好"
    assert checked > 0, "没匹配到任何'优化前后'的句子, 正则可能过期了"


def test_level_percentages_are_high_and_improvements_are_moderate():
    """绝对水平和改进幅度不能用同一个数值池.

    「Lighthouse 评分达到 30 分」是差评、「命中率提升至 50%」等于没提升 ——
    两者语义不同, 合理区间也不同.
    """
    import re

    level = re.compile(r"(?:达到|提升至) (\d+)%")
    for role in TRACKS:
        for seed in range(30):
            text = resume_text(random_resume(random.Random(seed), role=role))
            for value in level.findall(text):
                assert int(value) >= 80, f"{role}/{seed}: 绝对水平 {value}% 太低, 不像'提升后'"


def test_metrics_match_the_track():
    """指标要跟方向对得上.

    之前给"数据看板系统（个人项目）"配了「检索 MRR 达到 0.995」——
    一个看板项目哪来的检索指标? 这类错配会让人觉得整份简历是乱编的.
    """
    for seed in range(20):
        backend = resume_text(random_resume(random.Random(seed), role="后端开发"))
        assert "MRR" not in backend, "后端方向不该出现检索指标"
        assert "Lighthouse" not in backend

        frontend = resume_text(random_resume(random.Random(seed), role="前端开发"))
        assert "MRR" not in frontend


# --------------------------------------------------------------------------- #
# 交付物本身
# --------------------------------------------------------------------------- #
def test_every_track_produces_expected_sections():
    """每份简历都要有这五节 —— 面试官 Agent 的提纲是围绕它们规划的."""
    expected = ["教育背景", "专业技能", "实习经历", "项目经历", "自我评价"]
    for role in TRACKS:
        content = random_resume(random.Random(3), role=role)
        titles = [title for title, _ in content.sections]
        assert titles == expected, f"{role} 的章节不对: {titles}"


def test_resume_keeps_probe_targets():
    """随机化**内容**可以, 但不能把"可被追问的结构"随机掉.

    一份通顺但无懈可击的简历 demo 起来是没戏的 —— 面试官找不到下手的地方,
    只能问"介绍一下你的项目". 所以每份都必须带着那几类槽点.
    """
    for role in TRACKS:
        text = resume_text(random_resume(random.Random(9), role=role))
        # 量化数字(但没给口径)
        assert "%" in text or "QPS" in text or "MRR" in text, f"{role}: 没有任何量化数字"
        # 模糊职责
        assert "负责" in text or "参与" in text
        # 大词自我评价
        assert "学习能力强" in text or "团队协作" in text or "自驱力" in text
    # 且要告诉演示者靶子在哪
    assert len(random_resume(random.Random(9)).probes) >= 3


def test_same_seed_reproduces_same_resume():
    """同一种子必须产出同一份 —— "刚才那份挺好, 再来一份"要能复现."""
    a = resume_text(random_resume(random.Random(2026)))
    b = resume_text(random_resume(random.Random(2026)))
    assert a == b
