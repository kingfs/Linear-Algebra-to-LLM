#!/usr/bin/env node
/**
 * 检查 dist 内所有 <a href>：站内页面、锚点、lab/figures 静态文件。
 * 用法：pnpm build && node scripts/check-site-links.mjs
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DIST = path.resolve(__dirname, "../dist");
const rawBase = process.env.BASE_PATH || "/";
const SITE_BASE = rawBase.endsWith("/") ? rawBase : rawBase + "/";
if (!fs.existsSync(DIST)) {
  console.error("dist 不存在，先运行 pnpm build");
  process.exit(1);
}

function walk(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const abs = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(abs, out);
    else if (entry.name.endsWith(".html")) out.push(abs);
  }
  return out;
}

const files = walk(DIST);
const docs = new Map();
for (const file of files) {
  const html = fs.readFileSync(file, "utf8");
  const hrefs = [...html.matchAll(/<a\b[^>]*\bhref="([^"]*)"/g)].map((m) => m[1]);
  const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]));
  docs.set(file, { hrefs, ids });
}

const bad = [];
let anchors = 0;
for (const [file, doc] of docs) {
  const rel = path.relative(DIST, file);
  const base = "/" + path.dirname(rel).split(path.sep).filter(Boolean).join("/") + "/";
  for (const href of doc.hrefs) {
    if (!href || /^(#|https?:|mailto:|tel:|javascript:)/.test(href)) continue;
    const url = new URL(href, `https://site.invalid${base}`);
    let pathname = decodeURIComponent(url.pathname);
    if (SITE_BASE !== "/" && pathname.startsWith(SITE_BASE)) {
      pathname = "/" + pathname.slice(SITE_BASE.length);
    }
    let target = path.join(DIST, pathname);
    if (url.pathname.endsWith("/") || fs.existsSync(target) && fs.statSync(target).isDirectory()) {
      target = path.join(target, "index.html");
    } else if (!path.extname(target)) {
      target = path.join(target, "index.html");
    }
    if (!fs.existsSync(target)) {
      bad.push(`${rel}: 断链 -> ${href}`);
      continue;
    }
    if (url.hash) {
      anchors++;
      const id = decodeURIComponent(url.hash.slice(1));
      const dest = docs.get(target);
      if (dest && !dest.ids.has(id)) bad.push(`${rel}: 锚点不存在 -> ${href}`);
    }
  }
}

if (bad.length) {
  for (const line of bad) console.error("ERROR", line);
  console.error(`\n✗ ${files.length} 页，${bad.length} 个断链/锚点`);
  process.exit(1);
}
console.log(`✓ ${files.length} 页，${anchors} 个锚点，0 断链`);
