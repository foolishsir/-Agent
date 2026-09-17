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
