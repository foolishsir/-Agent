/**
 * 用**前端自己的 encodeWav/resample 代码**在 Node 里造一个 16kHz WAV.
 *
 * 为什么要这么做
 * --------------
 * 浏览器的录音路径(WAV 16kHz)和之前验证过的路径(edge-tts 的 24kHz mp3)
 * 在服务端走的是**不同的分支**: format 映射不同、采样率探测不同.
 * 而我没法驱动浏览器, 所以退而求其次 ——
 * 直接从前端 HTML 里**切出真实的函数**来跑, 保证产出和浏览器一模一样.
 *
 * (同样的手法项目里已经在用: frontend-tests/check-markdown.js 也是
 *  从 HTML 里切出 renderMarkdown 来跑用例, 而不是复制一份实现.)
 *
 * 输出: data/browser_shape.wav —— 16kHz 单声道 PCM16, 带 44 字节头.
 */

"use strict";

const fs = require("fs");
const path = require("path");

const HTML = path.join(__dirname, "..", "backend", "app", "static", "index.html");
const html = fs.readFileSync(HTML, "utf8");
const js = html.match(/<script>([\s\S]*?)<\/script>/)[1];

// 从页面里切出真实的实现 —— 而不是抄一份, 抄的那份迟早和页面对不上
function extract(name) {
  const re = new RegExp(`function ${name}\\([\\s\\S]*?\\n\\}`, "m");
  const m = js.match(re);
  if (!m) throw new Error(`没找到函数 ${name}`);
  return m[0];
}

const src = [extract("resample"), extract("encodeWav"), "module.exports = { resample, encodeWav };"].join("\n\n");
const mod = { exports: {} };
new Function("module", "exports", src)(mod, mod.exports);
const { resample, encodeWav } = mod.exports;

// --- 造一段"像语音"的信号 ---
// 单纯的正弦波 ASR 会返回空(正确行为), 这里的目的只是验证**容器与格式**
// 能被服务端正确接受, 所以用一段带包络的多频信号就够.
const SRC_RATE = 48000; // 模拟真实设备常见的 48k, 强制走重采样分支
const SECONDS = 2.0;
const n = Math.floor(SRC_RATE * SECONDS);
const samples = new Float32Array(n);
for (let i = 0; i < n; i++) {
  const t = i / SRC_RATE;
  const env = Math.sin((Math.PI * t) / SECONDS); // 淡入淡出, 避免爆音
  samples[i] =
    0.3 * env * Math.sin(2 * Math.PI * 180 * t) +
    0.15 * env * Math.sin(2 * Math.PI * 320 * t);
}

const resampled = resample(samples, SRC_RATE, 16000);
const wav = encodeWav(resampled, 16000);
const buf = Buffer.from(wav);

// --- 自检: 头必须完全符合 WAV 规范 ---
const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
const s = (off, len) => buf.toString("ascii", off, off + len);
const checks = [
  ["RIFF 魔数", s(0, 4) === "RIFF"],
  ["WAVE 标识", s(8, 4) === "WAVE"],
  ["fmt 块", s(12, 4) === "fmt "],
  ["PCM 格式", view.getUint16(20, true) === 1],
  ["单声道", view.getUint16(22, true) === 1],
  ["采样率 = 16000", view.getUint32(24, true) === 16000],
  ["字节率 = 32000", view.getUint32(28, true) === 32000],
  ["块对齐 = 2", view.getUint16(32, true) === 2],
  ["位深 = 16", view.getUint16(34, true) === 16],
  ["data 块", s(36, 4) === "data"],
  ["RIFF 长度自洽", view.getUint32(4, true) === buf.length - 8],
  ["data 长度自洽", view.getUint32(40, true) === resampled.length * 2],
  ["采样点数量 = 32000", resampled.length === 32000],
];

let bad = 0;
for (const [name, ok] of checks) {
  if (!ok) bad++;
  console.log(`  ${ok ? "ok  " : "FAIL"} ${name}`);
}

const out = path.join(__dirname, "..", "data", "browser_shape.wav");
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, buf);
console.log(`\n输入 ${SRC_RATE}Hz → 重采样到 16000Hz, ${resampled.length} 个采样点`);
console.log(`已写出: ${out} (${buf.length} 字节)`);
console.log(`\n结果: ${checks.length - bad} 通过, ${bad} 失败`);
process.exit(bad ? 1 : 0);
