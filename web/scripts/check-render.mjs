#!/usr/bin/env node
/**
 * 内容渲染检查：专门抓「Markdown 能通过、浏览器里却显示错」的问题。
 *
 * 检查项：
 * 1. 表格单元格里的 `|`：`$...$` 内含裸 `|` 会把表格切碎，公式退化成原始字符串；
 * 2. `<details>` 后缺空行，会把后面的 `## 10. 一页速记` 吞进 HTML 块；
 * 3. 公式定界符 `$` 单行不成对；
 * 4. 概念页里残留 ASCII 字符画（应该改成 SVG / Mermaid / 表格）；
 * 5. 若已构建 dist：正文出现未渲染的 `$`、或 KaTeX 报错。
 *
 * 用法：
 *   node scripts/check-render.mjs
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.resolve(__dirname, "..");
const ROOT = path.resolve(WEB, "..");
const DIST = path.join(WEB, "dist");

const errors = [];
const warnings = [];
let scanned = 0;

function walk(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === "node_modules" || entry.name === "dist" || entry.name === ".git") continue;
    const abs = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(abs, out);
    else if (entry.name.endsWith(".md")) out.push(abs);
  }
  return out;
}

const files = ["docs", "cheatsheets", "exercises", "curriculum"]
  .map((d) => path.join(ROOT, d))
  .filter((d) => fs.existsSync(d))
  .flatMap((d) => walk(d));
for (const entry of fs.readdirSync(ROOT, { withFileTypes: true })) {
  if (entry.isDirectory() && /^\d{2}-/.test(entry.name)) walk(path.join(ROOT, entry.name), files);
}
for (const f of ["README.md", "AGENTS.md", "PROGRESS.md"]) {
  if (fs.existsSync(path.join(ROOT, f))) files.push(path.join(ROOT, f));
}

const ART_LINE = /[─│┌┐└┘├┤┬┴┼]|→|←|↑|↓|--+>|\+--|\|__/;

for (const abs of files) {
  const rel = path.relative(ROOT, abs).replaceAll(path.sep, "/");
  const isConcept = /^\d{2}-[^/]+\/\d{2}-.+\.md$/.test(rel);
  const lines = fs.readFileSync(abs, "utf8").split("\n");
  let inFence = false;
  let fenceLang = "";
  let artLines = [];
  scanned++;

  lines.forEach((line, i) => {
    const fence = line.match(/^```(\w*)/);
    if (fence) {
      if (!inFence) {
        inFence = true;
        fenceLang = fence[1] || "";
        artLines = [];
      } else {
        if (isConcept && (fenceLang === "" || fenceLang === "text") && artLines.length >= 2) {
          errors.push(`${rel}:${i + 1} ASCII 字符画（${artLines.length} 行），改用 SVG / Mermaid / 表格`);
        }
        inFence = false;
        fenceLang = "";
      }
      return;
    }
    if (inFence) {
      if (ART_LINE.test(line)) artLines.push(i + 1);
      return;
    }

    if (/>/.test(line) && /<\/details>\s*$/.test(line) && i + 1 < lines.length && lines[i + 1].trim()) {
      errors.push(`${rel}:${i + 1} </details> 后缺空行，下一段会被当成 HTML 文本`);
    }
    if (/<details/.test(line) && i > 0 && lines[i - 1].trim()) {
      errors.push(`${rel}:${i + 1} <details> 前缺空行`);
    }

    if (line.trim().startsWith("|")) {
      for (const m of line.matchAll(/\$([^$\n]+)\$/g)) {
        if (/(?<!\\)\|/.test(m[1])) {
          errors.push(`${rel}:${i + 1} 表格公式内含裸 |，会切碎单元格：$${m[1].slice(0, 40)}…`);
        }
      }
    }

    const bare = line.replace(/\\\$/g, "").replace(/\$\$/g, "");
    if (bare.split("$").length % 2 === 0 && !/^\s*>?\s*(```|#)/.test(line)) {
      errors.push(`${rel}:${i + 1} 公式定界符 $ 不成对`);
    }
  });
}

// ---------------------------------------------------------------- dist 检查
let distChecked = 0;
if (fs.existsSync(DIST)) {
  const htmls = walkHtml(DIST);
  for (const file of htmls) {
    const html = fs.readFileSync(file, "utf8");
    const page = path.relative(DIST, file).replaceAll(path.sep, "/");
    distChecked++;
    if (html.includes("katex-error")) errors.push(`dist/${page}: KaTeX 解析失败（katex-error）`);
    for (const block of html.match(/<div class="mermaid">[\s\S]*?<\/div>/g) || []) {
      if (block.includes("<pre")) {
        errors.push(`dist/${page}: Mermaid 源码被空行截断（div 内出现 <pre>），图表会渲染成语法错误`);
      }
    }
    const text = html
      .replace(/<script[\s\S]*?<\/script>/g, " ")
      .replace(/<style[\s\S]*?<\/style>/g, " ")
      .replace(/<pre[\s\S]*?<\/pre>/g, " ")
      .replace(/<code[\s\S]*?<\/code>/g, " ")
      .replace(/<[^>]+>/g, " ");
    if (text.includes("$")) {
      const snippet = text.slice(Math.max(0, text.indexOf("$") - 30), text.indexOf("$") + 60).replace(/\s+/g, " ");
      errors.push(`dist/${page}: 正文残留未渲染的 $：「${snippet}」`);
    }
  }
} else {
  warnings.push("dist 不存在，跳过渲染后检查（先运行 pnpm build）");
}

function walkHtml(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const abs = path.join(dir, entry.name);
    if (entry.isDirectory()) walkHtml(abs, out);
    else if (entry.name.endsWith(".html")) out.push(abs);
  }
  return out;
}

for (const w of warnings) console.log(`  ! ${w}`);
for (const e of errors) console.log(`  ✗ ${e}`);
console.log(
  `\n检查 ${scanned} 个 Markdown 文件` +
    (distChecked ? `、${distChecked} 个 HTML 页面` : "") +
    `：${errors.length} error，${warnings.length} warning`
);
process.exit(errors.length ? 1 : 0);
