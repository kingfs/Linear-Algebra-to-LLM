# Web 学习平台

这是「从线性代数到大模型」课程的 Web 层：阅读、随堂测、知识卡、错题本与学习进度。内容仍然来自仓库根目录的 Markdown，Web 不保存第二份正文。

## 技术栈

- **Astro 7**：静态生成 174 个页面，正文默认零 JS。
- **React islands**：只有题目、卡片、进度这些交互组件按需 hydrate。
- **KaTeX + remark-math**：构建时渲染数学公式。
- **Mermaid**：把 Markdown 里的流程图转成 SVG。
- **IndexedDB**：阅读进度、答题记录、卡片排期全部本地保存，无后端。
- **Python + Matplotlib**：预生成解释型 SVG 图，见 `figures/`。

## 快速开始

根目录已提供 go-task 入口，见 [`../docs/本地开发.md`](../docs/本地开发.md)：

```bash
task setup
task dev
```

等价的手动命令：

```bash
cd web
pnpm install
pnpm dev --port 4321
# http://127.0.0.1:4321/
```

生产构建：

```bash
pnpm build      # 先 sync，再 astro build，输出 web/dist
pnpm preview    # 预览 dist
```

## 常用命令

| 命令 | 作用 |
|---|---|
| `pnpm sync` | 读取根目录 Markdown，生成 `src/content/generated`、`src/data`、`public/search.json`、`public/figures` |
| `pnpm dev` | sync + Astro 开发服务器 |
| `pnpm build` | sync + 静态构建 |
| `pnpm check` | sync + `astro check`（TypeScript/Astro 类型检查） |
| `node scripts/e2e.mjs` | 用本机 Chrome 跑 13 项端到端检查 |
| `pnpm links` | 检查构建后 174 页的站内链接与锚点 |
| `node scripts/check-quiz.mjs` | 校验题库字段、概念 id、source 与锚点 |

## 内容管线

```text
根目录 93 篇 Markdown
        │  scripts/sync-content.mjs
        ├─→ src/content/generated/*.md   Astro 内容集合
        ├─→ src/data/nav.json            侧边栏、上下篇
        ├─→ src/data/search.json         搜索索引
        ├─→ src/data/quizbank.json       按概念聚合题目
        ├─→ src/data/cards.json          从「一页速记」抽出的知识卡
        └─→ public/search.json           浏览器搜索用
```

**不要手改以下生成目录**：

- `src/content/generated/`
- `src/data/`
- `public/search.json`
- `public/figures/`

它们已在 `.gitignore` 中。改正文请改根目录 Markdown，然后 `pnpm sync`。

### 链接改写

`sync-content.mjs` 会把文档里的相对 `.md` 链接改写成站内路由：

- 概念页 → `/learn/<module>/<slug>/`
- 模块 README → `/modules/<module>/`
- 习题/答案 → `/pages/exercises/.../`
- 其它 docs/cheatsheets → `/pages/.../`

新增 Markdown 时不需要手动改链接，只要目标文件在仓库内。

## 页面

| 路由 | 说明 |
|---|---|
| `/` | 首页、路线、模块入口 |
| `/learn/m01/dot-product/` | 概念页：正文、图、目录、随堂测、上下篇 |
| `/modules/m01/` | 模块导览 |
| `/practice/` | 练习场，按模块组卷 |
| `/practice/m01/` | 整模块练习 |
| `/review/` | 知识卡与间隔重复 |
| `/wrong/` | 错题本 |
| `/progress/` | 阅读/正确率/卡片/薄弱点 |
| `/pages/...` | 习题、速查表、学习地图等附录 |

## 题库

题目在 `../quizbank/mNN.yaml`，规范见 [`../quizbank/README.md`](../quizbank/README.md)。

```bash
cd web
node scripts/check-quiz.mjs ../quizbank/m01.yaml
```

每道题要求：

- `concept` 必须是真实概念 id；
- `source` 指向真实 Markdown，并带 `#小节锚点`；
- 数值题答案必须可复算，带 `tolerance`；
- 判断题 `answer` 是布尔值；
- 选择题答案下标必须在选项范围内。

构建时 `sync-content.mjs` 会再校验一遍；任何错误会让构建失败。

## 图

`../figures/scripts/*.py` 用 Matplotlib 生成 `../figures/out/*.svg`，`../figures/manifest.yaml` 把图映射到概念 id。概念页会自动在正文前插入第一张匹配且未标记 `inline: true` 的图；正文内联图只由 Markdown 引用。

重新生成：

```bash
uv run --with matplotlib --with numpy python figures/scripts/fig_projection.py
```

或在仓库根目录批量：

```bash
for f in figures/scripts/fig_*.py; do uv run --with matplotlib --with numpy python "$f"; done
```

## 数据与隐私

- 阅读进度：IndexedDB store `progress`
- 答题记录：`attempts`
- 卡片排期：`cards`
- 无账号、无网络请求、无埋点。
- `/progress/` 可导出 JSON 备份；换设备时导入功能尚未实现（后续可加）。

## 端到端检查

```bash
cd web
pnpm build
pnpm links     # 163 页 / 97 个锚点 / 0 断链
node scripts/e2e.mjs
```

使用本机 Chrome，检查：

1. 概念页题目加载；
2. 数值题判分与解析；
3. 选择题提交；
4. 结果页与答题记录；
5. 标记已读并刷新后保留；
6. 搜索返回结果；
7. 知识卡翻面与排期；
8. 模块练习页；
9. 进度页读取本地数据；
10. 生成 SVG 图加载；
11. Mermaid 渲染为 SVG；
12. 移动端侧边栏。

## 部署

`web/dist` 是纯静态目录，可以直接放到 GitHub Pages、Cloudflare Pages、Netlify 或任意静态服务器。

### 根路径（本地、用户主页、自定义域名根）

```bash
cd web
pnpm build
```

默认 `BASE_PATH=/`，输出 `web/dist`，直接部署即可。

### 子路径（如 `https://yongzhen.wang/Linear-Algebra-to-LLM/`）

```bash
cd web
BASE_PATH=/Linear-Algebra-to-LLM/ pnpm build
BASE_PATH=/Linear-Algebra-to-LLM/ pnpm links
```

- `astro.config.mjs` 会把 `BASE_PATH` 传给 Astro 的 `base`；
- `scripts/prefix-base.mjs` 会给静态 HTML 里的根路径链接加前缀；
- React island 里的运行时链接通过 `src/lib/base.ts` 的 `withBase()` 加前缀；
- 搜索请求 `/search.json` 也会跟随 `BASE_URL`。

### GitHub Pages（tag 触发）

仓库带 `.github/workflows/deploy-pages.yml`：

- 只在 push `v*` tag（或手动 `workflow_dispatch`）时构建、部署；
- 默认 `BASE_PATH=/Linear-Algebra-to-LLM/`；
- 如果自定义域名把项目站挂在根路径，在仓库 **Settings → Secrets and variables → Actions → Variables** 添加 `PAGES_BASE_PATH=/`，无需改代码；
- GitHub Pages 的 Source 选 **GitHub Actions**。

> 注意：GitHub Pages 的自定义域名可能把项目站挂在域名根路径（`https://yongzhen.wang/`），而不是 `/Linear-Algebra-to-LLM/`。若实际访问是根路径，把 `PAGES_BASE_PATH` 改成 `/`；若确实要挂在 `/Linear-Algebra-to-LLM/`，需要由反向代理或托管平台提供该子路径。

## 已知取舍

- 题库为 406 题（每篇 3–7 题），以概念理解为主，不追求题海。
- 知识卡由「一页速记」自动抽取，部分卡片偏“填空”，后续可手工精修。
- 搜索是前端子串匹配，中文可用但不如专业分词；93 篇规模足够。
- 全白化/PCA 图用静态 SVG，交互组件后续再加（2D 变换、SVD 分步、KV 计算器）。
