#!/usr/bin/env node
/**
 * 把仓库根目录的 Markdown 同步为 Astro 内容集合，并生成导航、搜索、题库、卡片与图的数据。
 *
 * 设计原则：根目录 Markdown 是唯一源；src/content/generated 是构建产物，不入库。
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import matter from "gray-matter";
import YAML from "yaml";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.resolve(__dirname, "..");
const ROOT = path.resolve(WEB, "..");
const GENERATED = path.join(WEB, "src/content/generated");
const DATA = path.join(WEB, "src/data");
const PUBLIC_FIG = path.join(WEB, "public/figures");
const PUBLIC_LABS = path.join(WEB, "public/labs");

const MODULES = {
  M00: "准备：数学语言",
  M01: "向量与几何",
  M02: "向量空间",
  M03: "矩阵与线性映射",
  M04: "线性方程组",
  M05: "行列式",
  M06: "特征值",
  M07: "SVD",
  M08: "微积分与优化",
  M09: "概率与统计",
  M10: "数值线性代数",
  M11: "大模型专题",
  M12: "进阶与全景",
};

const warnings = [];
const errors = [];

function rmrf(p) {
  fs.rmSync(p, { recursive: true, force: true });
}
function ensure(p) {
  fs.mkdirSync(p, { recursive: true });
}
function writeJSON(file, value) {
  fs.writeFileSync(path.join(DATA, file), JSON.stringify(value, null, 2) + "\n");
}
function stripMd(text) {
  return text
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/<div class="mermaid">[\s\S]*?<\/div>/g, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/[>#*_`$]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}
function firstH1(body) {
  const m = body.match(/^#\s+(.+)$/m);
  return m ? m[1].trim() : "";
}
function slugifyRoute(s) {
  return s.replace(/[?#]+$/g, "").replace(/\/+/g, "/");
}

// ---------------------------------------------------------------- 收集源文件
const sourceFiles = [];
for (const entry of fs.readdirSync(ROOT, { withFileTypes: true })) {
  if (!entry.isDirectory()) continue;
  const m = entry.name.match(/^(\d{2})-(.+)$/);
  if (!m) continue;
  const module = "M" + m[1];
  const dir = path.join(ROOT, entry.name);
  for (const f of fs.readdirSync(dir)) {
    if (!f.endsWith(".md")) continue;
    sourceFiles.push({ abs: path.join(dir, f), module, moduleDir: entry.name, type: f === "README.md" ? "module" : "concept" });
  }
}
for (const sub of ["exercises", "docs", "cheatsheets", "curriculum", "labs"]) {
  const dir = path.join(ROOT, sub);
  if (!fs.existsSync(dir)) continue;
  for (const f of fs.readdirSync(dir)) {
    if (!f.endsWith(".md")) continue;
    sourceFiles.push({ abs: path.join(dir, f), module: "附录", moduleDir: sub, type: sub === "exercises" ? "exercise" : "page" });
  }
}
for (const f of ["README.md", "AGENTS.md", "PROGRESS.md", "quizbank/README.md", "web/README.md"]) {
  const abs = path.join(ROOT, f);
  if (fs.existsSync(abs)) sourceFiles.push({ abs, module: "附录", moduleDir: "", type: "page" });
}

// ---------------------------------------------------------------- 解析与路由
const docs = [];
const routeBySource = new Map();

for (const src of sourceFiles) {
  const raw = fs.readFileSync(src.abs, "utf8");
  const parsed = matter(raw);
  const rel = path.relative(ROOT, src.abs).replaceAll(path.sep, "/");
  const data = parsed.data || {};
  let route, kind, id, title, order, minutes, level, tags, llm_hooks, slug;

  if (src.type === "concept") {
    slug = String(data.slug || path.basename(src.abs, ".md").replace(/^\d+-/, ""));
    id = String(data.id || "");
    route = `/learn/${src.module.toLowerCase()}/${slug}/`;
    kind = "concept";
    title = data.title || firstH1(parsed.content) || slug;
    order = Number(String(id).split(".").pop() || 0) || 0;
    minutes = Number(data.minutes || 30);
    level = data.level || "核心";
    tags = data.tags || [];
    llm_hooks = data.llm_hooks || [];
  } else if (src.type === "module") {
    slug = `module-${src.module.toLowerCase()}`;
    id = `module-${src.module}`;
    route = `/modules/${src.module.toLowerCase()}/`;
    kind = "module";
    title = firstH1(parsed.content) || `${src.module} ${MODULES[src.module] || ""}`;
    order = Number(src.module.slice(1));
    minutes = 20;
    level = "核心";
    tags = ["模块导览"];
    llm_hooks = [];
  } else {
    const relNoExt = rel.replace(/\.md$/, "");
    slug = rel === "README.md" ? "home" : relNoExt.replace(/[\/]/g, "__");
    id = `page-${slug}`;
    route = rel === "README.md" ? "/pages/home/" : `/pages/${relNoExt}/`;
    kind = src.type;
    title = firstH1(parsed.content) || relNoExt;
    order = 0;
    minutes = 10;
    level = "选学";
    tags = [];
    llm_hooks = [];
  }

  routeBySource.set(path.resolve(src.abs), route);
  docs.push({
    abs: src.abs,
    rel,
    body: parsed.content,
    data: { ...data, id, title, module: src.module, slug, route, kind, order, minutes, level, tags, llm_hooks },
    moduleDir: src.moduleDir,
    type: src.type,
  });
}

// ---------------------------------------------------------------- 链接改写
const linkRe = /\]\(([^)\s]+)(\s+"[^"]*")?\)/g;
function rewriteLinks(body, srcAbs) {
  if (path.basename(srcAbs) === "模板-概念文档.md") {
    return body
      .replace(/\[([^\]]+)\]\(01-xxx\.md\)/g, "$1")
      .replace(/\[([^\]]+)\]\(……\)/g, "$1");
  }
  return body.replace(linkRe, (full, target, title = "") => {
    if (/^(https?:|mailto:|tel:|#)/.test(target)) return full;
    if (!/\.md(?:#.*)?$/.test(target) && !target.startsWith(".") && !target.startsWith("/")) return full;
    const [rawPath, anchor] = target.split("#");
    if (rawPath.endsWith(".py")) {
      const abs = path.resolve(path.dirname(srcAbs), rawPath);
      if (fs.existsSync(abs) && path.dirname(abs) === path.join(ROOT, "labs")) {
        return `](/labs/${encodeURIComponent(path.basename(abs))}${anchor ? "#" + anchor : ""}${title})`;
      }
      return full;
    }
    if (!rawPath.endsWith(".md")) return full;
    const abs = path.resolve(path.dirname(srcAbs), rawPath);
    const route = routeBySource.get(abs);
    if (!route) {
      warnings.push(`未映射链接 ${path.relative(ROOT, srcAbs)} -> ${target}`);
      return full;
    }
    return `](${route}${anchor ? "#" + anchor : ""}${title})`;
  });
}

function transformMermaid(body) {
  return body.replace(/```mermaid\n([\s\S]*?)```/g, (_m, code) => `<div class="mermaid">\n${code.trim()}\n</div>`);
}

// ---------------------------------------------------------------- 写内容
ensure(GENERATED);
ensure(DATA);
ensure(PUBLIC_FIG);
ensure(PUBLIC_LABS);
rmrf(GENERATED);
rmrf(DATA);
ensure(GENERATED);
ensure(DATA);

const navModules = new Map();
for (const doc of docs) {
  let body = rewriteLinks(doc.body, doc.abs);
  body = transformMermaid(body);
  const fm = {
    id: doc.data.id,
    title: doc.data.title,
    module: doc.data.module,
    slug: doc.data.slug,
    route: doc.data.route,
    kind: doc.data.kind,
    order: doc.data.order,
    minutes: doc.data.minutes,
    level: doc.data.level,
    tags: doc.data.tags,
    llm_hooks: doc.data.llm_hooks,
    source: doc.rel,
    moduleTitle: MODULES[doc.data.module] || "附录",
  };
  const filename =
    doc.type === "concept"
      ? `${doc.data.module.toLowerCase()}--${doc.data.slug}.md`
      : doc.type === "module"
        ? `${doc.data.module.toLowerCase()}--module.md`
        : `page--${doc.data.slug}.md`;
  fs.writeFileSync(path.join(GENERATED, filename), `---\n${YAML.stringify(fm).trim()}\n---\n\n${body}`);

  if (doc.type === "concept" || doc.type === "module") {
    const m = doc.data.module;
    if (!navModules.has(m)) navModules.set(m, { id: m, title: `${m} ${MODULES[m] || ""}`.trim(), route: `/modules/${m.toLowerCase()}/`, concepts: [] });
    if (doc.type === "concept") {
      navModules.get(m).concepts.push({
        id: doc.data.id,
        title: doc.data.title,
        route: doc.data.route,
        slug: doc.data.slug,
        minutes: doc.data.minutes,
        level: doc.data.level,
        order: doc.data.order,
      });
    }
  }
}

const nav = [...navModules.values()].sort((a, b) => a.id.localeCompare(b.id));
for (const m of nav) m.concepts.sort((a, b) => a.order - b.order);
const conceptDocs = docs.filter((d) => d.type === "concept");

// ---------------------------------------------------------------- 搜索索引
const search = conceptDocs.map((doc) => {
  const sectionText = doc.body
    .split(/^##\s+/m)
    .slice(1)
    .map((s) => stripMd(s).slice(0, 500))
    .join(" ");
  return {
    id: doc.data.id,
    title: doc.data.title,
    module: doc.data.module,
    route: doc.data.route,
    level: doc.data.level,
    tags: doc.data.tags,
    text: `${stripMd(doc.body).slice(0, 1800)} ${sectionText}`.slice(0, 2600),
  };
});

// ---------------------------------------------------------------- 知识卡：从「一页速记」抽取
const cards = [];
for (const doc of conceptDocs) {
  const m = doc.body.match(/^##\s*10\.[^\n]*\n([\s\S]*)/m);
  if (!m) continue;
  const section = m[1].split(/^##\s/m)[0];
  const lines = section
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => /^[-*]\s+|^\d+\.\s+/.test(l))
    .map((l) => l.replace(/^[-*]\s+|^\d+\.\s+/, "").trim());
  lines.forEach((line, i) => {
    if (line.length < 8) return;
    const bold = line.match(/\*\*([^*]+)\*\*/);
    let front;
    if (bold) {
      front = line.replace(bold[0], "____");
    } else {
      const cut = line.split(/[：:。]/)[0];
      front = cut.length >= 6 ? `${cut}……` : line.slice(0, 12) + "……";
    }
    cards.push({
      id: `card-${doc.data.id}-${i + 1}`,
      concept: doc.data.id,
      title: doc.data.title,
      route: doc.data.route,
      front: stripMd(front),
      back: stripMd(line),
    });
  });
}

// ---------------------------------------------------------------- 题库
const quizByConcept = {};
const quizQuestions = [];
const quizFiles = fs.existsSync(path.join(ROOT, "quizbank"))
  ? fs.readdirSync(path.join(ROOT, "quizbank")).filter((f) => f.endsWith(".yaml") || f.endsWith(".yml"))
  : [];
const knownIds = new Set(conceptDocs.map((d) => String(d.data.id)));
for (const file of quizFiles) {
  let bank;
  try {
    bank = YAML.parse(fs.readFileSync(path.join(ROOT, "quizbank", file), "utf8"));
  } catch (e) {
    errors.push(`${file}: YAML 解析失败 ${e.message}`);
    continue;
  }
  for (const q of bank.questions || []) {
    const where = `${file}:${q.id || "(无 id)"}`;
    if (!q.id) errors.push(`${where}: 缺少 id`);
    if (!knownIds.has(String(q.concept))) errors.push(`${where}: concept ${q.concept} 不存在`);
    if (!["single_choice", "multiple_choice", "true_false", "numeric"].includes(q.type)) {
      errors.push(`${where}: 未知题型 ${q.type}`);
    }
    if (!q.explanation) errors.push(`${where}: 缺少 explanation`);
    if (!q.source) errors.push(`${where}: 缺少 source`);
    if (q.type === "single_choice" || q.type === "multiple_choice") {
      if (!Array.isArray(q.options) || q.options.length < 2) errors.push(`${where}: options 至少 2 个`);
      if (!Array.isArray(q.answer) || !q.answer.length) errors.push(`${where}: answer 必须是数组`);
    }
    if (q.type === "numeric" && typeof q.answer !== "number") errors.push(`${where}: numeric answer 必须是数字`);
    let sourceRoute;
    if (q.source) {
      const [srcPath, anchor] = String(q.source).split("#");
      const abs = path.resolve(ROOT, srcPath);
      const route = routeBySource.get(abs);
      if (route) sourceRoute = route + (anchor ? `#${anchor}` : "");
      else warnings.push(`${where}: source 未映射 ${q.source}`);
    }
    const item = { ...q, module: bank.module, concept: String(q.concept), sourceRoute };
    quizQuestions.push(item);
    (quizByConcept[item.concept] ||= []).push(item);
  }
}

// ---------------------------------------------------------------- 图
let figures = [];
const figManifestPath = path.join(ROOT, "figures/manifest.yaml");
if (fs.existsSync(figManifestPath)) {
  try {
    const manifest = YAML.parse(fs.readFileSync(figManifestPath, "utf8"));
    figures = (manifest.figures || []).filter((f) => {
      const src = path.join(ROOT, "figures/out", f.file);
      if (!fs.existsSync(src)) {
        warnings.push(`图缺失：figures/out/${f.file}`);
        return false;
      }
      fs.copyFileSync(src, path.join(PUBLIC_FIG, f.file));
      return true;
    });
  } catch (e) {
    errors.push(`figures/manifest.yaml 解析失败：${e.message}`);
  }
}

// ---------------------------------------------------------------- 输出数据
writeJSON("nav.json", nav);
writeJSON("search.json", search);
writeJSON("cards.json", cards);
writeJSON("quizbank.json", quizByConcept);
writeJSON("figures.json", figures);

// 把 labs 的 .py 作为静态资源提供，方便网页里的实验链接直接打开
rmrf(PUBLIC_LABS);
ensure(PUBLIC_LABS);
const labsDir = path.join(ROOT, "labs");
if (fs.existsSync(labsDir)) {
  for (const f of fs.readdirSync(labsDir)) {
    if (f.endsWith(".py")) fs.copyFileSync(path.join(labsDir, f), path.join(PUBLIC_LABS, f));
  }
}
writeJSON("stats.json", {
  concepts: conceptDocs.length,
  modules: nav.length,
  questions: quizQuestions.length,
  cards: cards.length,
  figures: figures.length,
  generatedAt: new Date().toISOString(),
});

fs.writeFileSync(path.join(WEB, "public/search.json"), JSON.stringify(search));
console.log(`✓ 同步 ${conceptDocs.length} 篇概念 / ${nav.length} 个模块`);
console.log(`✓ 题库 ${quizQuestions.length} 题，知识卡 ${cards.length} 张，图 ${figures.length} 张`);
if (warnings.length) {
  console.log(`⚠ ${warnings.length} 条警告`);
  for (const w of warnings.slice(0, 12)) console.log("  -", w);
}
if (errors.length) {
  console.error(`✗ ${errors.length} 条错误`);
  for (const e of errors.slice(0, 30)) console.error("  -", e);
  process.exit(1);
}
