# 动手实验（Labs）

> 数学不是看会的，是算会、跑会的。每个 lab 用 100–230 行 NumPy 验证一个核心直觉，并附「你应该看到什么」。

## 环境

需要 Python 3.10+ 与 NumPy；`requirements.txt` 中的 Matplotlib 仅用于可选画图，所有 lab 都能纯文本运行。

```bash
cd labs
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python3 lab01_相似度与投影.py
```

也可以直接用 `uv`：

```bash
cd labs
uv run --with numpy python lab01_相似度与投影.py
```

## 实验清单

| Lab | 对应章节 | 验证什么 |
|---|---|---|
| lab01 相似度与投影 | 1.3–1.6 | 点积/余弦/投影的几何关系 |
| lab02 基与秩 | 2.3–2.6 | 线性无关、秩、Gram–Schmidt、相似变换 |
| lab03 矩阵四视角 | 3.1–3.3 | 行视角/列视角/映射视角一致 |
| lab04 最小二乘 | 4.4–4.6 | 正规方程、投影残差、条件数 |
| lab05 行列式 | 5.1–5.4 | 面积/体积缩放、det(AB)、Jacobian |
| lab06 特征值与幂迭代 | 6.1–6.6 | 特征方向、对角化、收敛速度、PageRank |
| lab07 SVD 与低秩 | 7.1–7.4 | 奇异值衰减、截断误差、伪逆 |
| lab08 梯度与反向传播 | 8.1–8.5 | 数值梯度校验、GD、手写反向传播 |
| lab09 PCA 与马尔可夫 | 9.3–9.5 | 主成分、白化、稳态分布 |
| lab10 数值误差与 GEMM | 10.1–10.3 | 机器精度、条件数、矩阵乘耗时 |
| lab11 Transformer 形状链 | 11.1–11.11 | embedding→attention→KV Cache→LoRA→采样 |
| lab12 einsum 与随机矩阵 | 12.1–12.2 | einsum 等价性、谱范数、MP 律 |

## 约定

- 每个脚本**自包含**：不依赖仓库其它模块，复制出去也能跑。
- 输出用中文打印关键结论；随机数固定 `np.random.seed(...)`，保证可复现。
- 脚本末尾用注释给出「思考题」，与对应章节练习呼应。
- 如果某一步与你手算不一致，先信手算，再检查代码索引（0/1 起点）。
