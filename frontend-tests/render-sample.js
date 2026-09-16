/**
 * 用真实回答验证渲染效果(开发用).
 *
 * 用法: node frontend-tests/render-sample.js <回答文本文件>
 */

const fs = require("fs");
const path = require("path");

const html = fs.readFileSync(
  path.join(__dirname, "..", "backend", "app", "static", "index.html"),
  "utf8"
);

const escSrc = html.slice(html.indexOf("const esc ="), html.indexOf("const STATUS_LABEL"));
const renderSrc = html.slice(
  html.indexOf("function renderCiteBadges"),
  html.indexOf("/** 流式渲染器")
);
const { renderMarkdown } = new Function(escSrc + renderSrc + "return { renderMarkdown };")();

const file = process.argv[2];
const src = file ? fs.readFileSync(file, "utf8") : "示例 **加粗** 与 [1]";
const out = renderMarkdown(src);

const count = (s) => (out.match(s) || []).length;

console.log("原始 Markdown 长度:", src.length);
console.log("渲染后 HTML 长度 :", out.length);
console.log();
console.log("=== 结构检查 ===");
console.log("  <ul> 列表      :", count(/<ul>/g), "个");
console.log("  <li> 列表项    :", count(/<li>/g), "个");
console.log("  <strong> 加粗  :", count(/<strong>/g), "处");
console.log("  <p> 段落       :", count(/<p>/g), "个");
console.log("  引用角标       :", count(/cite-ref/g), "个");
console.log("  残留的 **      :", count(/\*\*/g), "处  (应为 0)");
console.log("  残留的 - 列表号:", count(/(?:\s|^)- /g), "处  (应为 0)");
console.log();
console.log("=== 渲染输出前 600 字符 ===");
console.log(out.slice(0, 600));
