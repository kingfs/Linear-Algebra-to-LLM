# 动手实验（Labs）

> 数学不是看会的，是算会、跑会的。每个 lab 用 30–60 行 NumPy 验证一个核心直觉，并附「你应该看到什么」。

## 环境

需要 Python 3.10+ 与 NumPy（可选 Matplotlib 画图）。

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
uv run --with numpy --with matplotlib python lab01_相似度与投影.py
```

## 实验清单

| Lab | 对应章节 | 验证什么 |
|---|---|---|
| lab01 相似度与投影 | 1.3–1.6 | 点积/余弦/投影的几何关系 |
| lab02 基与秩 | 2.3–2.6 | 线性无关、秩、Gram–Schmidt |
| lab03 矩阵四视角 | 3.1–3.3 | 行视角/列视角/映射视角一致 |
| lab04 最小二乘 | 4.4 | 正规方程与正交投影 |
| lab05 行列式 | 5.1–5.2 | 面积缩放与可逆性 |
| lab06 特征值与幂迭代 | 6.1–6.6 | 特征方向、收敛速度 |
| lab07 SVD 与低秩 | 7.1–7.3 | 奇异值衰减、低秩近似误差 |
| lab08 梯度与反向传播 | 8.1–8.5 | 数值梯度校验 |
| lab09 PCA | 9.3 | 特征分解 = 最大方差方向 |
| lab10 数值稳定性 | 10.1–10.2 | 条件数放大误差 |
| lab11 Attention 形状 | 11.3–11.4 | 单头 attention 全流程 |

## 约定

- 每个脚本**自包含**：不依赖仓库其它模块，复制出去也能跑。
- 输出用中文打印关键结论；随机数固定 `np.random.seed(...)`，保证可复现。
- 脚本末尾用注释给出「思考题」，与对应章节练习呼应。
- 如果某一步与你手算不一致，先信手算，再检查代码索引（0/1 起点）。
