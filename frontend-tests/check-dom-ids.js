/**
 * 前端 DOM 引用自检.
 *
 * 为什么需要这个测试
 * ------------------
 * 单文件前端最容易踩的坑不是逻辑写错, 而是**在顶层事件绑定里写错一个 id**:
 *
 *     $("iv-sumbit").addEventListener(...)   // 拼错了
 *
 * 后果是 `$()` 返回 null, 抛 TypeError, **整个 script 块中断执行** ——
 * 页面看起来"加载出来了"(HTML 是静态的), 但所有按钮都没反应,
 * 而且控制台之外没有任何提示.
 *
 * 这类错误浏览器不会在构建期报(本来也没有构建期), 人工点页面也未必
 * 每个按钮都点到. 所以用一个静态检查兜住:
 *
 *   1. JS 里所有 $("xxx") 引用的 id 必须在 HTML 里存在
 *   2. 每个 data-view 必须有对应的 view-xxx 容器
 *   3. 反向也查一遍: HTML 里定义了但 JS 从没用过的 id(可能是改名后的残留)
 *
 * 用法: node frontend-tests/check-dom-ids.js
 */

"use strict";

const fs = require("fs");
const path = require("path");

const FILE = path.join(__dirname, "..", "backend", "app", "static", "index.html");
const html = fs.readFileSync(FILE, "utf8");

const ids = new Set(
  Array.from(html.matchAll(/\bid="([^"]+)"/g), (m) => m[1]),
);

const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/);
if (!scriptMatch) {
  console.error("✗ 没找到 <script> 块");
  process.exit(1);
}
const js = scriptMatch[1];

// $("xxx") 引用
const used = new Set(
  Array.from(js.matchAll(/\$\("([^"]+)"\)/g), (m) => m[1]),
);
// getElementById("xxx") 引用
for (const m of js.matchAll(/getElementById\("([^"]+)"\)/g)) used.add(m[1]);

const failures = [];
const checks = [];

// ---------------- 1. JS 引用的 id 必须存在 ----------------
const missing = Array.from(used).filter((id) => !ids.has(id)).sort();
checks.push({
  name: "JS 引用的 id 都已定义",
  ok: missing.length === 0,
  detail: missing.length ? `缺失: ${missing.join(", ")}` : `${used.size} 个引用`,
});

// ---------------- 2. data-view 必须有对应容器 ----------------
const views = Array.from(html.matchAll(/data-view="([^"]+)"/g), (m) => m[1]);
const badViews = views.filter((v) => !ids.has("view-" + v));
checks.push({
  name: "每个 data-view 都有 view-* 容器",
  ok: badViews.length === 0,
  detail: badViews.length ? `缺失: ${badViews.join(", ")}` : views.join(" / "),
});

// ---------------- 3. 顶层事件绑定不能绑到不存在的 id ----------------
// 顶层(非函数内)的 $("x").addEventListener 一旦绑到 null 会直接中断整个脚本
const topLevelBind = Array.from(
  js.matchAll(/^\$\("([^"]+)"\)\.addEventListener/gm),
  (m) => m[1],
);
const deadBinds = topLevelBind.filter((id) => !ids.has(id));
checks.push({
  name: "顶层事件绑定的 id 都已定义",
  ok: deadBinds.length === 0,
  detail: deadBinds.length
    ? `会导致脚本中断: ${deadBinds.join(", ")}`
    : `${topLevelBind.length} 处绑定`,
});

// ---------------- 4. 反向: 有没有定义了却没人用的 id ----------------
// 这条只作为提示, 不算失败 —— 有些 id 是给 CSS 或用户脚本用的
const unused = Array.from(ids)
  .filter((id) => !used.has(id) && !js.includes(`"${id}"`) && !js.includes(`'${id}'`))
  .filter((id) => !id.startsWith("view-") && !id.startsWith("health"))
  .sort();

// ---------------- 5. 不许把「瞬时全局状态」烙进 disabled ----------------
//
// 这一条来自一个真实且很隐蔽的 bug:
//
//   renderDocs() 里写了 `${busy ? "disabled" : ""}`, 而 renderDocs 是在
//   upload() 的 `await loadDocs()` 中调用的 —— 那一刻 busy 还是 true
//   (finally 还没执行)。于是**每次上传完成, 整张列表的删除按钮全变禁用**,
//   而之后没有任何东西会重渲染它(切换页签当时也不加载列表)。
//
//   结果: 按钮点下去没有请求、没有报错、没有任何反馈 ——
//   用户只能猜"是不是必须留一个文档?"。
//
// 根因是**渲染期读取了一个会独立于本次渲染而变化的全局状态**:
// 状态恢复了, 界面却已经永久停在错误显示上。
//
// 注意和 renderSkills 里 `${full ? "disabled" : ""}` 的区别 ——
// 那个**不算违规**: `full` 是由 IV.skillIds 现算出来的派生值,
// 而每次勾选变化都会重新调用 renderSkills, 状态和渲染是同步的, 卡不住。
//
// 所以规则精确到"瞬时的全局标志", 而不是"凡是用到 disabled 就报"。
const TRANSIENT_GLOBALS = ["busy", "loading", "saving", "uploading", "submitting", "pending"];
const bakedState = [];
const bakedRe = /\$\{\s*(\w+)\s*(?:\?|&&)[^}]*?["']disabled["'][^}]*?\}/g;
for (const m of html.matchAll(bakedRe)) {
  if (!TRANSIENT_GLOBALS.includes(m[1])) continue;
  const line = html.slice(0, m.index).split("\n").length;
  bakedState.push(`第 ${line} 行: ${m[0].trim()}`);
}
checks.push({
  name: "渲染时不把瞬时状态烙进 disabled",
  ok: bakedState.length === 0,
  detail: bakedState.length
    ? `会导致按钮永久失效: ${bakedState.join(" | ")}`
    : `检查了 ${TRANSIENT_GLOBALS.length} 个瞬时状态变量`,
});

// ---------------- 输出 ----------------
let failed = 0;
for (const c of checks) {
  if (!c.ok) failed++;
  console.log(`  ${c.ok ? "ok   " : "FAIL "} ${c.name} — ${c.detail}`);
}
if (unused.length) {
  console.log(`  note  可能是残留的 id(${unused.length} 个): ${unused.join(", ")}`);
}

console.log(`\n结果: ${checks.length - failed} 通过, ${failed} 失败`);
process.exit(failed ? 1 : 0);
