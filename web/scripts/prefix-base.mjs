#!/usr/bin/env node
/**
 * 构建后给静态 HTML 的根路径链接加部署前缀。
 *
 * 站点正文里的链接由 Markdown 同步生成，天然是 /learn/... 这种根路径；
 * Astro 的 base 只影响它自己生成的资源，不会改写这些链接，所以需要这一层。
 *
 * 用法：
 *   BASE_PATH=/LinearAlgebra/ pnpm build
 *   BASE_PATH=/ pnpm build          # 默认，无操作
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DIST = path.resolve(__dirname, "../dist");
const raw = process.env.BASE_PATH || "/";
const base = raw.endsWith("/") ? raw : raw + "/";
const prefix = base.replace(/\/$/, "");

if (base === "/") {
  console.log("BASE_PATH=/，跳过路径前缀");
  process.exit(0);
}
if (!fs.existsSync(DIST)) {
  console.error("dist 不存在，先运行 astro build");
  process.exit(1);
}

function walk(dir, ext, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const abs = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(abs, ext, out);
    else if (entry.name.endsWith(ext)) out.push(abs);
  }
  return out;
}

let pages = 0;
let attrs = 0;
for (const file of walk(DIST, ".html")) {
  const before = fs.readFileSync(file, "utf8");
  const after = before
    .replace(/(\b(?:href|src|action|poster)=)(["'])(\/(?!\/)[^"']*)\2/g, (match, key, quote, value) => {
      if (value === prefix || value.startsWith(prefix + "/")) return match;
      attrs++;
      return `${key}${quote}${prefix}${value}${quote}`;
    })
    .replace(/(\bsrcset=)(["'])([^"']*)\2/g, (match, key, quote, value) => {
      const next = value
        .split(",")
        .map((part) => {
          const [url, ...rest] = part.trim().split(/\s+/);
          if (!url || !url.startsWith("/") || url.startsWith("//")) return part.trim();
          if (url === prefix || url.startsWith(prefix + "/")) return part.trim();
          attrs++;
          return [prefix + url, ...rest].join(" ");
        })
        .join(", ");
      return `${key}${quote}${next}${quote}`;
    });
  if (after !== before) {
    fs.writeFileSync(file, after);
    pages++;
  }
}

// CSS 里的 @font-face 也会写死 /_astro/...，必须一起加前缀
let cssFiles = 0;
let cssUrls = 0;
for (const file of walk(DIST, ".css")) {
  const before = fs.readFileSync(file, "utf8");
  const after = before.replace(/url\(\s*(["']?)(\/(?!\/)[^)"']+)\1\s*\)/g, (match, quote, value) => {
    if (value === prefix || value.startsWith(prefix + "/")) return match;
    cssUrls++;
    return `url(${quote}${prefix}${value}${quote})`;
  });
  if (after !== before) {
    fs.writeFileSync(file, after);
    cssFiles++;
  }
}

console.log(`✓ base=${base}：${pages} 页，${attrs} 个根路径属性，${cssFiles} 个 CSS 文件（${cssUrls} 个 url）已加前缀`);
