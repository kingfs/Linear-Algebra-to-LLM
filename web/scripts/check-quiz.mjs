#!/usr/bin/env node
/** 独立校验 quizbank/*.yaml，不写任何生成文件；供并行写题时使用。 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import matter from "gray-matter";
import YAML from "yaml";
import GithubSlugger from "github-slugger";

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ROOT = path.resolve(WEB, "..");
const args = process.argv.slice(2);
const files = args.length
  ? args.map((f) => path.resolve(process.cwd(), f))
  : fs.readdirSync(path.join(ROOT, "quizbank")).filter((f) => /\.ya?ml$/.test(f)).map((f) => path.join(ROOT, "quizbank", f));

const concepts = new Map();
for (const dir of fs.readdirSync(ROOT, { withFileTypes: true })) {
  if (!dir.isDirectory()) continue;
  const m = dir.name.match(/^(\d{2})-(.+)$/);
  if (!m) continue;
  for (const f of fs.readdirSync(path.join(ROOT, dir.name))) {
    if (!f.endsWith(".md") || f === "README.md") continue;
    const raw = fs.readFileSync(path.join(ROOT, dir.name, f), "utf8");
    const data = matter(raw).data;
    if (!data.id) continue;
    const slugger = new GithubSlugger();
    const anchors = new Set();
    for (const line of raw.split("\n")) {
      const m = line.match(/^##\s+(.+)$/);
      if (m) anchors.add(slugger.slug(m[1].trim()));
    }
    concepts.set(String(data.id), { module: data.module, file: path.join(dir.name, f), title: data.title, anchors });
  }
}

const errors = [];
const warnings = [];
const allIds = new Set();
let total = 0;

for (const file of files) {
  const rel = path.relative(ROOT, file);
  let bank;
  try {
    bank = YAML.parse(fs.readFileSync(file, "utf8"));
  } catch (e) {
    errors.push(`${rel}: YAML 解析失败：${e.message}`);
    continue;
  }
  if (!bank || !Array.isArray(bank.questions)) {
    errors.push(`${rel}: 缺少 questions 数组`);
    continue;
  }
  const counts = new Map();
  for (const q of bank.questions) {
    total++;
    const where = `${rel}:${q.id || "(无 id)"}`;
    if (!q.id) errors.push(`${where}: 缺少 id`);
    else if (allIds.has(q.id)) errors.push(`${where}: id 重复`);
    allIds.add(q.id);
    if (!q.concept) errors.push(`${where}: 缺少 concept`);
    const c = concepts.get(String(q.concept));
    if (!c) errors.push(`${where}: concept ${q.concept} 不存在`);
    else if (c.module !== bank.module) errors.push(`${where}: concept ${q.concept} 属于 ${c.module}，不属于 ${bank.module}`);
    if (!Number.isInteger(q.section) || q.section < 0 || q.section > 10) errors.push(`${where}: section 必须是 0–10 的整数`);
    if (!["single_choice", "multiple_choice", "true_false", "numeric"].includes(q.type)) errors.push(`${where}: 未知题型 ${q.type}`);
    if (!q.stem || String(q.stem).trim().length < 8) errors.push(`${where}: stem 太短`);
    if (!q.explanation || String(q.explanation).trim().length < 8) errors.push(`${where}: explanation 太短`);
    if (!q.source) errors.push(`${where}: 缺少 source`);
    else {
      const [src, anchor] = String(q.source).split("#");
      if (!fs.existsSync(path.join(ROOT, src))) errors.push(`${where}: source 文件不存在 ${src}`);
      if (!/\.md$/.test(src)) errors.push(`${where}: source 必须指向 .md`);
      if (!anchor) warnings.push(`${where}: source 没有 #小节锚点`);
      else if (c && !c.anchors.has(anchor)) warnings.push(`${where}: 锚点 #${anchor} 在 ${src} 中不存在`);
    }
    if (q.type === "single_choice" || q.type === "multiple_choice") {
      if (!Array.isArray(q.options) || q.options.length < 2) errors.push(`${where}: options 至少 2 个`);
      else if (new Set(q.options.map((x) => String(x).trim())).size !== q.options.length) errors.push(`${where}: options 有重复`);
      if (!Array.isArray(q.answer) || q.answer.length === 0) errors.push(`${where}: answer 必须是数组`);
      else {
        for (const a of q.answer) if (!Number.isInteger(a) || a < 0 || a >= q.options.length) errors.push(`${where}: answer 下标 ${a} 越界`);
        if (q.type === "single_choice" && q.answer.length !== 1) errors.push(`${where}: 单选题 answer 必须只有 1 个`);
      }
      if (q.options?.some((x) => /以上都|都不对|all of the above/i.test(String(x)))) errors.push(`${where}: 不要使用「以上都」类选项`);
    } else if (q.type === "true_false") {
      if (typeof q.answer !== "boolean") errors.push(`${where}: 判断题 answer 必须是 true/false`);
    } else if (q.type === "numeric") {
      if (typeof q.answer !== "number" || !Number.isFinite(q.answer)) errors.push(`${where}: 数值题 answer 必须是有限数字`);
      if (q.tolerance !== undefined && (typeof q.tolerance !== "number" || q.tolerance <= 0)) errors.push(`${where}: tolerance 必须为正数`);
    }
    counts.set(String(q.concept), (counts.get(String(q.concept)) || 0) + 1);
  }
  const moduleConcepts = [...concepts.entries()].filter(([, c]) => c.module === bank.module).map(([id]) => id);
  for (const id of moduleConcepts) {
    const n = counts.get(id) || 0;
    if (n === 0) warnings.push(`${rel}: 概念 ${id} 没有题目`);
    else if (n < 3) warnings.push(`${rel}: 概念 ${id} 只有 ${n} 题（建议 ≥3）`);
  }
}

console.log(`检查 ${files.length} 个文件，${total} 道题`);
if (warnings.length) {
  console.log(`⚠ ${warnings.length} 条警告`);
  for (const w of warnings.slice(0, 40)) console.log("  -", w);
}
if (errors.length) {
  console.error(`✗ ${errors.length} 条错误`);
  for (const e of errors.slice(0, 60)) console.error("  -", e);
  process.exit(1);
}
