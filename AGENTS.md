# AGENTS.md — 本仓库的 Agent 协作指南

> 这是一份写给「未来的自己 / 其他 agent / 人类贡献者」的操作手册。
> 开始任何任务前，先读完本文件的 **§0 一分钟速览** 与 **§11 工作流**。
> 最后更新：2026-10-23 ｜ 维护者：课程作者（人类）与 AI agent 共同维护。

---

## 0. 一分钟速览

- **这是什么**：一套面向「想从底层理解大模型」的学习者的线性代数课程。中文、结构化、由浅入深、概念独立成篇。
- **写作第一原则**：**先直觉，再定义，再推导，最后落到大模型**。不写「定义—定理—证明」的冷冰冰教科书。
- **每个概念一个文件**，放在 `NN-模块名/` 下，文件名 `NN-概念名.md`。
- **完成的定义**：概念文档通过 `python3 scripts/check.py`，模块 `README.md` 与习题已更新，`PROGRESS.md` 状态已改，然后 **本地 git commit**。
- **不要**：编造论文/实验数据；留 `TODO` 占位符；跳过「通往大模型」小节；一次性重写别人已完成的模块。
- **当前进度**：见 [`PROGRESS.md`](PROGRESS.md)。**下一个任务**永远从那里挑。

---

## 1. 项目目标与非目标

### 目标
1. 覆盖线性代数**完整概念体系**：向量、空间、矩阵、方程组、行列式、特征值、SVD、二次型、数值稳定性，以及为大模型服务的微积分/概率/优化接口。
2. 每个概念回答四个问题：
   - 它**直觉上**是什么？（几何图像 / 物理类比 / 数据操作）
   - 它**为什么**被这样定义？（没有它会缺什么）
   - 它**怎么算**？举一个**场景化**的例子（推荐系统、图像、词向量、注意力……）。
   - 它在大模型里**出现在哪**？（embedding、attention、LoRA、量化、KV Cache……）
3. 对 agent 友好：结构可预测、元数据可解析、检查可自动化、任务可增量认领。

### 非目标
- 不是考研题海；证明追求「讲清楚为什么」，而非形式化完备。
- 不是 PyTorch 教程；代码只用于**验证直觉**。
- 不追新闻；大模型关联以**稳定原理**为主（矩阵形状、计算图、低秩、谱）。

---

## 2. 目录结构

```
.
├── AGENTS.md                 # 本文件：协作宪法
├── README.md                 # 课程首页：地图、路线、使用方法
├── PROGRESS.md               # 模块进度与待办（agent 的任务队列）
├── docs/                     # 全局说明：学习地图、符号表、模板
├── curriculum/               # 大纲与周计划
├── 00-准备/ ... 12-进阶/      # 课程主体，一个模块一个目录
│   ├── README.md             # 模块导览：目标、概念地图、顺序、自测
│   └── NN-概念名.md           # 概念文档（模板见 docs/模板-概念文档.md）
├── exercises/                # 每模块习题与答案（mNN-习题.md / mNN-答案.md）
├── labs/                     # 可运行的 NumPy 实验，验证直觉
├── cheatsheets/              # 公式速查、符号表、术语中英对照
├── quizbank/                 # Web 题库（mNN.yaml，按概念组织）
├── figures/                  # Matplotlib 图脚本、SVG 产物与 manifest
├── web/                      # Astro + React islands 学习平台
└── scripts/                  # check.py 等质量门禁
```

模块编号与主题：

| 编号 | 目录 | 主题 | 大模型钩子 |
|---|---|---|---|
| M00 | `00-准备/` | 数学语言、张量轴的观点 | 形状 `[B,T,D]` |
| M01 | `01-向量与几何/` | 向量、点积、范数、投影 | embedding、余弦相似度 |
| M02 | `02-向量空间/` | 空间、无关、基、秩、正交化 | 表示空间、维度 |
| M03 | `03-矩阵与线性映射/` | 矩阵四视角、乘法、逆、LU、四子空间 | 线性层、权重矩阵 |
| M04 | `04-线性方程组/` | 解的结构、最小二乘、伪逆、条件数 | 欠定/过定、微调 |
| M05 | `05-行列式/` | 体积、可逆性、Jacobian | 归一化流、变量替换 |
| M06 | `06-特征值/` | 特征分解、谱定理、二次型、幂迭代 | 稳定性、PageRank |
| M07 | `07-SVD/` | 奇异值、低秩近似、伪逆 | LoRA、模型压缩 |
| M08 | `08-微积分与优化/` | 梯度、Jacobian、Hessian、反向传播 | 训练、梯度 |
| M09 | `09-概率与统计/` | 协方差、高斯、PCA、马尔可夫 | 表征几何、采样 |
| M10 | `10-数值线性代数/` | 浮点、条件数、GEMM、迭代法 | GPU、显存、精度 |
| M11 | `11-大模型专题/` | Attention、RoPE、LayerNorm、KV Cache、量化、MoE | 端到端串讲 |
| M12 | `12-进阶/` | 张量/einsum、随机矩阵、全景图 | 推理研究路线 |

---

## 3. 编号、命名与元数据

- 概念文件：`NN-概念名.md`，`NN` 两位，从 `01` 开始，**不跳号**。
- 模块 `README.md` 中的章节号（如 `1.3`）与文件名解耦：文件名管排序，`id` 管引用。
- 所有概念文档必须有 YAML front matter：

```yaml
---
id: 1.3                    # 章节号，全局唯一，字符串
title: 点积：把相似变成数字
module: M01                # 模块编号
slug: dot-product          # ASCII 短名，用于锚点/脚本
prereq: [1.1, 1.2]         # 前置概念 id，可为 []
minutes: 30                # 预计阅读+练习分钟
level: 核心                # 核心 | 进阶 | 选学
tags: [点积, 相似度, 投影]
llm_hooks: [embedding, attention]
---
```

- 修改 `id` / `slug` 时必须全局搜索并修复引用。

---

## 4. 概念文档的固定结构

完整模板见 [`docs/模板-概念文档.md`](docs/模板-概念文档.md)。**十一个二级标题（0–10），顺序固定，不可缺**：

| 小节 | 必须回答的问题 |
|---|---|
| `## 0. 场景引入` | 一个具体、有画面感的问题（不要一上来就定义） |
| `## 1. 直觉` | 用几何/物理/数据操作建立心理图像 |
| `## 2. 定义` | 精确的数学定义 + 每个符号的解释 |
| `## 3. 为什么这样定义` | 去掉它会失去什么？与相邻概念如何衔接？ |
| `## 4. 性质与推导` | 关键性质、一条核心推导（不必是全套证明） |
| `## 5. 例子` | 至少一个**手算数值例子** + 一个场景例子 |
| `## 6. 常见误区` | 2–4 个真实易错点，说明错在哪 |
| `## 7. 概念网络` | 与前置/后续概念的关系，一句话各自 |
| `## 8. 通往大模型` | 具体到形状/公式/代码位置，不空谈 |
| `## 9. 练习` | 3–5 题：概念题、计算题、开放题各至少一题 |
| `## 10. 一页速记` | 5 行以内可背诵的要点 |

允许在各节内使用小标题（`###`），但**不要新增顶级 `##`**。

---

## 5. 写作风格（重要）

1. **中文为主**，术语首次出现给英文：`特征值（eigenvalue）`。之后可只用中文。
2. **一段只讲一件事**，段落 ≤ 5 行；多用短句、列表、表格。
3. **先问问题，再给答案**。每节开头尽量用一个问句或场景把读者拉进来。
4. **公式必须被解释**：出现一个公式后，用一句话读出来（「这个式子说的是……」）。
5. **例子要具体**：给数字、给形状、给单位。禁止「设 $A$ 为任意矩阵」了事。
6. **类比要标注边界**：类比之后补一句「这个类比在哪里失效」。
7. **不堆砌定理**。一个概念最多 1–2 个核心定理，其余放「进阶阅读」链接。
8. **大模型关联要可验证**：写出矩阵形状、参与运算的项、量级；不确定就写「在常见实现中」，不编造具体论文结论。
9. 语气：像一位耐心的导师在白板前讲解，可以用「你」，不用「笔者」。
10. 禁止空洞套话：「众所周知」「不言而喻」「显然」——要么解释，要么删掉。

---

## 6. 数学排版规范

- 行内公式 `$...$`，独立公式 `$$...$$`，前后空行。
- 标量：小写斜体 $a,\lambda$；向量：小写粗体 $\mathbf{x}$；矩阵：大写斜体 $A$；单位阵 $I_n$；零向量 $\mathbf{0}$。
- 转置 $A^\top$（不用 $A^T$）；逆 $A^{-1}$；伪逆 $A^{+}$；共轭转置 $A^*$。
- 范数 $\|\mathbf{x}\|_2$；内积 $\langle \mathbf{x},\mathbf{y}\rangle$；逐元素积 $\odot$；Kronecker 积 $\otimes$。
- 集合 $\mathbb{R}^n$、$\mathbb{R}^{m\times n}$；映射 $T:\mathbb{R}^n\to\mathbb{R}^m$。
- 分段函数用 `cases`；矩阵用 `bmatrix`；对齐用 `aligned`。
- 数字与单位之间留空格；中文与公式之间不加空格由排版器处理，但源码中保持可读。
- 矩阵形状统一写成 `[m, n]` 或 $m\times n$；张量形状写成 `[B, T, D]`。
- 索引从 1 开始讲数学，从 0 开始讲代码，必须显式说明。

---

## 7. 「通往大模型」小节的要求

这是本课程与普通教材的分水岭。每篇至少包含：

- **出现位置**：哪个结构/哪一步计算（如「注意力打分 $QK^\top$」）。
- **形状**：参与运算的张量维度，如 $Q\in\mathbb{R}^{T\times d_k}$。
- **为什么需要这个数学**：没有它，哪一步算不出来/不稳定/太贵。
- **一个可检验的结论**：如「$d_k$ 越大，点积方差越大，所以要除以 $\sqrt{d_k}$」。
- 可选：3–10 行伪代码或 NumPy 片段。

避免：只写「这在 NLP 中有广泛应用」这类空话。

---

## 8. 交叉引用与导航

- 站内链接用相对路径：`[点积](01-向量与几何/03-点积.md)`。
- 引用概念时用「第 1.3 节」并附链接，不重复解释。
- 每篇文末（`## 10` 之后）可加一行「下一篇 →」，但模块 README 才是导航的权威来源。
- 修改文件名后运行 `python3 scripts/check.py` 找断链。

---

## 9. 质量门禁

提交前必须运行：

```bash
python3 scripts/check.py
python3 scripts/check_links.py   # 全仓 README/习题/速查表链接
```

检查内容：
1. front matter 必填字段齐全、`id` 唯一；
2. 十一个固定小节（0–10）齐全且顺序正确；
3. 「通往大模型」非空（≥ 80 字）；
4. 站内相对链接可解析；
5. 无 `TODO` / `TBD` / `待补充` 占位符；
6. 文件编码 UTF-8；
7. `check_links.py` 扫描所有 Markdown，0 断链。

不通过不允许 commit。若确需临时占位，在 `PROGRESS.md` 登记为「草稿」，且不得标记模块完成。

---

## 10. 提交规范

- 语言：中文 subject，可带英文 scope。
- 格式：`<type>(<scope>): <subject>`
  - `feat(m01): 新增点积、范数与投影`
  - `docs(m03): 补充四个基本子空间`
  - `fix(links): 修复 SVD 章节断链`
  - `chore: 更新进度与检查脚本`
- **一个逻辑任务一个 commit**；模块完成时单独一个 `feat(mNN): 完成模块 ...`。
- 只提交本任务相关文件；不要 `git add -A` 扫入无关改动。
- 不 amend 已推送提交；本仓库当前仅本地，但仍保持历史清晰。

---

## 11. 工作流（每个任务照做）

1. **读**：`PROGRESS.md` → 选一个 `pending` 任务（优先当前里程碑）。
2. **查**：读该模块 `README.md`、前置概念、`docs/模板-概念文档.md`、`docs/符号表.md`。
3. **写**：一次只做一个模块/一组强相关概念，保持风格一致。
4. **证**：手算例子必须自己验算；代码片段在 `labs/` 跑通（`uv run` 或 `python3`）。
5. **检**：`python3 scripts/check.py`，修复所有 error。
6. **更**：更新模块 `README.md`、`exercises/mNN-*.md`、`PROGRESS.md` 状态。
7. **提交**：`git add <具体文件> && git commit -m "feat(mNN): ..."`。
8. **记**：在 `PROGRESS.md` 的「变更日志」追加一行（日期 + 摘要 + commit 短哈希可后补）。

---

## 12. 任务粒度建议

- 小任务：1–3 篇概念文档（或 1 个 lab + 对应习题）。
- 中任务：完成一个模块（文档 + README + 习题 + lab）。
- 大任务：跨模块专题（如「Attention 全解剖」），拆成多 commit。
- 每次会话结束前，必须让仓库处于「检查通过 + 已提交」状态。

---

## 13. 给未来 agent 的提醒

- 先读 [`docs/学习地图.md`](docs/学习地图.md) 建立全局观，再动手。
- 不要为了「覆盖率」灌水。宁可少一篇，不可多一篇空话。
- 遇到数学不确定：用具体 $2\times2$ 例子验证，或标注「此处需要复核」并登记到 `PROGRESS.md`，**不要猜**。
- 大模型技术演进快，涉及具体模型参数时写「以某类实现为例」，并给出可自行验证的形状推导。
- 保持 `PROGRESS.md` 诚实：没写完就是草稿，不要标完成。

---

## 14. Web 学习平台（`web/`）

Web 端是课程的第二消费层，**不是第二内容源**。根目录 Markdown 仍是唯一正文来源；`web/src/content/generated/`、`web/src/data/`、`web/public/figures/`、`web/public/search.json` 都是构建产物，已在 `.gitignore` 中，**不要手改**。

### 14.1 常用命令

```bash
cd web
pnpm install
pnpm dev --port 4321     # http://127.0.0.1:4321/
pnpm build               # 静态构建到 web/dist
pnpm check               # astro check（TypeScript）
node scripts/e2e.mjs     # 12 项浏览器端到端检查
node scripts/check-quiz.mjs ../quizbank/m01.yaml
node scripts/check-site-links.mjs   # 构建后全站链接/锚点检查
node scripts/check-render.mjs      # 表格公式 / details / ASCII 图 / 未渲染 $ 检查
```

### 14.2 改内容后的顺序

1. 改根目录 Markdown（概念、模块 README、习题等）。
2. `cd web && pnpm sync`：重新生成内容集合、导航、搜索、题库、卡片、图。
3. `pnpm build` 或 `pnpm dev` 验证。
4. 根目录仍要跑 `python3 scripts/check.py` 与 `python3 scripts/check_links.py`。

### 14.3 题库

- 文件：`quizbank/mNN.yaml`，规范见 [`quizbank/README.md`](quizbank/README.md)。
- 每题必须有 `concept`、`section`、`type`、`stem`、`explanation`、`source`；`source` 指向真实 Markdown 并带 `#小节锚点`。
- 数值题答案必须复算，带 `tolerance`；判断题 `answer` 用布尔值；选择题 `answer` 是下标数组。
- 写完必须跑 `node scripts/check-quiz.mjs ../quizbank/mNN.yaml`，0 error 0 warning 才提交。
- 不要运行 `pnpm sync` 或 `pnpm build` 来“验证”单文件题库；并行写题时它们会写生成目录，改用上面的独立校验脚本。

### 14.4 图

- 脚本：`figures/scripts/fig_*.py`；产物：`figures/out/*.svg`；映射：`figures/manifest.yaml`。
- 重新生成：`uv run --with matplotlib --with numpy python figures/scripts/fig_projection.py`。
- 概念页按 `concepts` 匹配第一张图；新增图后在 manifest 里登记 `caption` 与 `alt`。
- 正文内联图：把 SVG 放进 `figures/out/`，概念页用相对路径引用（`../figures/out/xxx.svg`）；`pnpm sync` 会复制到 `web/public/figures/` 并改写成 `/figures/xxx.svg`，不要手改生成目录。
- **禁止 ASCII 字符画**：几何/数据图用 SVG，流程与关系用 Mermaid，表格性内容（矩阵行、对照表）用 Markdown 表格。表格单元格里的公式若含绝对值/范数，用 `\lvert … \rvert`、`\lVert … \rVert`，不要写裸 `|`（会把单元格切断）。

### 14.5 提交门禁

Web 相关改动至少满足：

```bash
cd web && pnpm build && node scripts/check-site-links.mjs && node scripts/check-render.mjs && node scripts/e2e.mjs
python3 scripts/check.py
python3 scripts/check_links.py
```

提交时只加源文件：`web/package.json`、`web/pnpm-lock.yaml`、`web/src/**`、`web/scripts/**`、`web/public/favicon.svg`、`web/README.md`、`quizbank/**`、`figures/scripts/**`、`figures/out/**`、`figures/manifest.yaml`。不要提交 `web/node_modules/`、`web/dist/`、`web/src/data/` 等生成物。

### 14.6 子路径与 GitHub Pages

- 本地默认 `BASE_PATH=/`；部署到 `https://yongzhen.wang/LinearAlgebra/` 这类子路径时：
  ```bash
  cd web
  BASE_PATH=/LinearAlgebra/ pnpm build
  BASE_PATH=/LinearAlgebra/ pnpm links
  ```
- Astro 的 `base` 由 `BASE_PATH` 注入；`scripts/prefix-base.mjs` 负责给同步生成的 Markdown 链接加前缀；React 组件统一使用 `src/lib/base.ts` 的 `withBase()`。
- `.github/workflows/deploy-pages.yml` 只在 `v*` tag 或手动触发时部署，默认 `PAGES_BASE_PATH=/LinearAlgebra/`。
- 如果 Pages 自定义域名实际挂在根路径，把仓库变量 `PAGES_BASE_PATH` 改为 `/`。
- 部署前至少验证：
  ```bash
  cd web
  BASE_PATH=/LinearAlgebra/ pnpm build
  BASE_PATH=/LinearAlgebra/ pnpm links
  BASE_URL=http://127.0.0.1:4322/LinearAlgebra node scripts/e2e.mjs
  ```
