/**
 * Markdown 渲染器的验证脚本(开发用, 不参与构建).
 *
 * 为什么需要它: 渲染逻辑写在单文件 HTML 里, 没有构建步骤也没有浏览器测试环境.
 * 这里把渲染函数从页面中**切出来**在 Node 里跑真实用例 ——
 * 既验证了逻辑, 也保证测试的是页面里的那份真实代码, 而不是一份复制品.
 *
 * 用法: node frontend-tests/check-markdown.js
 */

const fs = require("fs");
const path = require("path");

const HTML = path.join(__dirname, "..", "backend", "app", "static", "index.html");
const html = fs.readFileSync(HTML, "utf8");

// ---- 从页面中切出渲染相关的代码段 ----
const start = html.indexOf("function renderCiteBadges");
const end = html.indexOf("function renderMarkdown");
const tail = html.indexOf("/** 流式渲染器");

if (start < 0 || end < 0 || tail < 0) {
  console.error("无法定位渲染函数, 页面结构可能变了");
  process.exit(1);
}

const escSrc = html.slice(html.indexOf("const esc ="), html.indexOf("const STATUS_LABEL"));
const renderSrc = html.slice(start, tail);

const factory = new Function(`
  ${escSrc}
  ${renderSrc}
  return { renderMarkdown };
`);
const { renderMarkdown } = factory();

// ---- 用例 ----
let passed = 0;
let failed = 0;

function check(name, input, expectations) {
  const out = renderMarkdown(input);
  const problems = [];
  for (const [desc, fn] of expectations) {
    if (!fn(out)) problems.push(`${desc}\n      实际输出: ${out}`);
  }
  if (problems.length) {
    failed++;
    console.log(`  FAIL  ${name}`);
    problems.forEach((p) => console.log(`        ${p}`));
  } else {
    passed++;
    console.log(`  ok    ${name}`);
  }
}

const has = (s) => (o) => o.includes(s);
const not = (s) => (o) => !o.includes(s);

console.log("Markdown 渲染用例:\n");

check("加粗", "这是**重点内容**。", [
  ["应生成 <strong>", has("<strong>重点内容</strong>")],
  ["不应残留星号", not("**")],
]);

check("无序列表", "技术栈包括：\n- Java\n- Spring Boot\n- Redis", [
  ["应生成 <ul>", has("<ul>")],
  ["应生成 3 个 <li>", (o) => (o.match(/<li>/g) || []).length === 3],
  ["不应残留学号", not("* "), not("- Java")],
]);

check("有序列表", "步骤如下：\n1. 上传文档\n2. 等待解析\n3. 开始提问", [
  ["应生成 <ol>", has("<ol>")],
  ["应有 3 项", (o) => (o.match(/<li>/g) || []).length === 3],
]);

check("行内代码", "使用 `pip install` 安装依赖", [
  ["应生成 <code>", has("<code>pip install</code>")],
  ["反引号应消失", not("`")],
]);

check("代码块", "示例：\n```python\nprint('hi')\n```\n结束", [
  ["应生成 <pre><code>", has("<pre")],
  ["应保留语言标记", has('data-lang="python"')],
  ["内容应在 pre 内而非行内 code", has("<pre data-lang=\"python\"><code>print")],
  ["空白行应被吃掉", (o) => !o.includes("```")],
  ["占位符必须被还原", not("\u0000CB")],
]);

check("代码块内的星号不被当成加粗", "```\na = 2 ** 3\n```", [
  ["应保留原始星号", has("2 ** 3")],
  ["不应生成 <strong>", not("<strong>")],
]);

check("标题", "## 设备维护\n正文内容", [
  ["应生成 <h2>", has("<h2>设备维护</h2>")],
  ["不应残留井号", not("##")],
]);

check("引用块", "> 这是一段引用\n普通段落", [
  ["应生成 <blockquote>", has("<blockquote>")],
  ["不应残留 &gt;", not("&gt;")],
]);

check("引用角标", "钢刀寿命为 20000 次 [1]。锡膏需冷藏 [2][3]。", [
  ["应生成可点击角标", has('class="cite-ref" data-cite="1"')],
  ["应生成第二个角标", has('data-cite="2"')],
  ["方括号应消失", not("[1]")],
]);

check("整体列表 + 角标(真实场景)", "- Java [1][2][3]\n- Spring Boot [1][2]\n- Redis [3]", [
  ["应生成列表", has("<ul>")],
  ["应生成角标", has('data-cite="1"')],
  ["不应残留方括号", not("[1]")],
]);

check("XSS: 原始 HTML 标签必须被转义", '<script>alert(1)</script>', [
  ["不应出现可执行标签", not("<script>")],
  ["应被转义", has("&lt;script&gt;")],
]);

check("XSS: 图片 onerror 注入", '<img src=x onerror=alert(1)>', [
  ["不应生成 img 标签", not("<img")],
  ["应被转义", has("&lt;img")],
]);

check("XSS: 代码块内也不能逃逸", "```\n<script>alert(1)</script>\n```", [
  ["不应出现可执行标签", not("<script>")],
  ["应被转义", has("&lt;script&gt;")],
]);

check("链接", "参考 [官网](https://example.com) 说明", [
  ["应生成 <a>", has('href="https://example.com"')],
  ["应带 noopener", has("noopener")],
]);

check("链接与引用角标不冲突", "见 [1] 与 [文档](https://a.com)", [
  ["引用角标正常", has('data-cite="1"')],
  ["链接正常", has('href="https://a.com"')],
]);

check("表格", "| 参数 | 值 |\n|---|---|\n| 温度 | 0.1 |\n| 条数 | 5 |", [
  ["应生成 <table>", has('<table class="md-table">')],
  ["应有表头", has("<th>参数</th>")],
  ["应有数据行", has("<td>0.1</td>")],
]);

check("分割线", "上\n\n---\n\n下", [["应生成 <hr>", has("<hr>")]]);

check("段落与换行", "第一段。\n\n第二段。", [
  ["应生成两个段落", (o) => (o.match(/<p>/g) || []).length === 2],
]);

check("删除线", "旧值 ~~已废弃~~ 新值", [
  ["应生成 <del>", has("<del>已废弃</del>")],
]);

check("空输入", "", [["应返回空串", (o) => o === ""]]);

check("不做任何格式化的纯文本", "钢刀的更换周期是三个月。", [
  ["应包在 <p> 里", has("<p>")],
  ["内容应保留", has("钢刀的更换周期是三个月。")],
]);

console.log(`\n结果: ${passed} 通过, ${failed} 失败`);
process.exit(failed ? 1 : 0);
