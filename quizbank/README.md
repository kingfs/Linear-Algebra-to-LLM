# 题库编写规范

每道题必须能追溯到课程原文，并且答案可独立验证。这个目录是 Web 练习场的唯一题目源。

## 文件

按模块一个文件：`m00.yaml` … `m12.yaml`。顶层：

```yaml
module: M04
questions: []
```

## 题目字段

```yaml
- id: q-4.4-001              # 全局唯一；q-<concept>-<三位序号>
  concept: "4.4"             # 必须存在于对应模块文档 front matter
  section: 5                 # 题目主要考的小节 0–10
  type: numeric              # single_choice | multiple_choice | true_false | numeric
  difficulty: 2              # 1–3
  stem: "题干，可含 $...$"
  options: ["A", "B"]        # 选择题必填，至少 2 个
  answer: [0]                # 选择题：正确下标数组；单选只 1 个
  # true_false: answer: true
  # numeric: answer: 1.5, tolerance: 0.001, unit: ""
  explanation: "为什么；一两句话，可含 $...$"
  source: "04-线性方程组/04-最小二乘.md#5-例子"
  tags: [最小二乘, 手算]
```

## 出题优先级

1. **每篇至少 3 题**，覆盖不同小节：定义/性质、手算、误区或大模型关联。
2. 题型混合：每篇至少 1 道选择题或判断题；有明确数值例子的，至少 1 道数值题。
3. 判断题优先从 `## 6. 常见误区` 取材；把错误说法写成题干，解析引用纠正。
4. 数值题答案必须自己复算；小数给 `tolerance`（如 `0.01`）。
5. 单选题正确答案位置要分散，不能全是 A。
6. 解析末尾不必重复题干；给出关键一步即可。

## 校验

```bash
cd web && node scripts/check-quiz.mjs ../quizbank/m04.yaml
```

不要运行 `pnpm sync` 或 `pnpm build`（会写生成目录，可能与其他并行任务冲突）；最终由主流程统一同步。
