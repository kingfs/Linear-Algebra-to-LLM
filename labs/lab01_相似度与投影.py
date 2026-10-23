"""Lab01 · 相似度与投影

对应章节：1.3 点积 / 1.4 范数 / 1.5 余弦相似度 / 1.6 正交与投影

验证四件事：
1. 点积与夹角的几何关系；
2. 余弦相似度只看方向、不受长度影响；
3. 投影的残差与投影方向正交，且满足勾股定理；
4. attention 对 key 的范数敏感，而余弦不敏感。

运行：python3 lab01_相似度与投影.py
依赖：numpy
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)
np.random.seed(0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度（假设 a、b 非零）。"""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def softmax(z: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax。"""
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def entropy(p: np.ndarray) -> float:
    """自然对数熵，越大说明分布越均匀。"""
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# ---------------------------------------------------------------- 1
section("1. 点积与夹角")
x = np.array([3.0, 4.0])
y = np.array([4.0, 3.0])
dot = x @ y
cos_xy = cosine(x, y)
angle = np.degrees(np.arccos(np.clip(cos_xy, -1, 1)))
print(f"x = {x}, y = {y}")
print(f"点积 x·y = {dot}")
print(f"范数 ||x|| = {np.linalg.norm(x):.3f}, ||y|| = {np.linalg.norm(y):.3f}")
print(f"余弦 = {cos_xy:.3f}, 夹角 = {angle:.1f}°")
print(f"几何公式验证: ||x||·||y||·cosθ = {np.linalg.norm(x)*np.linalg.norm(y)*cos_xy:.3f}")


# ---------------------------------------------------------------- 2
section("2. 语义向量的余弦相似度矩阵")
# 每一维分别表示 [动物性, 机械性, 可食性]，仅用于演示
words = ["猫", "狗", "汽车", "苹果"]
E = np.array([
    [0.90, 0.05, 0.10],
    [0.85, 0.10, 0.15],
    [0.05, 0.95, 0.05],
    [0.10, 0.05, 0.90],
])

# 归一化后，余弦矩阵 = 单位向量矩阵 @ 其转置
En = E / np.linalg.norm(E, axis=1, keepdims=True)
S = En @ En.T

print("余弦相似度矩阵（行/列顺序：猫、狗、汽车、苹果）")
print(S)
print(f"\n猫-狗 = {S[0,1]:.3f}，猫-汽车 = {S[0,2]:.3f}，狗-汽车 = {S[1,2]:.3f}")
print("结论：语义相近的词方向更一致；余弦只看方向。")

# 长度不影响余弦
E_scaled = E.copy()
E_scaled[0] *= 7.5  # 把“猫”的向量放大 7.5 倍
cos_scaled = cosine(E_scaled[0], E[1])
print(f"\n把猫向量放大 7.5 倍后，猫-狗余弦 = {cos_scaled:.3f}（不变）")
print(f"但欧氏距离从 {np.linalg.norm(E[0]-E[1]):.3f} 变为 {np.linalg.norm(E_scaled[0]-E[1]):.3f}")


# ---------------------------------------------------------------- 3
section("3. 投影：残差正交与勾股定理")
v = np.array([3.0, 4.0])
u = np.array([1.0, 1.0])
alpha = (v @ u) / (u @ u)
p = alpha * u
e = v - p

print(f"v = {v}, u = {u}")
print(f"投影系数 α = (v·u)/(u·u) = {alpha:.3f}")
print(f"投影 p = {p}, 残差 e = {e}")
print(f"e·u = {e @ u:.3e}（应为 0）")
print(f"||v||² = {v @ v:.3f}, ||p||² + ||e||² = {p @ p + e @ e:.3f}")

P = np.outer(u, u) / (u @ u)
print(f"\n投影矩阵 P =\n{P}")
print(f"P² =\n{P @ P}（应等于 P）")
print(f"P·v = {P @ v}（应等于 p）")
print(f"P 的特征值 = {np.linalg.eigvalsh(P)}（应为 1 和 0）")


# ---------------------------------------------------------------- 4
section("4. attention：缩放与 key 范数的影响")
d_k = 64
q = np.random.randn(d_k)
k = np.random.randn(d_k)
scores_raw = np.array([q @ k, q @ (2 * k), q @ (-k)])
scores_scaled = scores_raw / np.sqrt(d_k)

for name, s in [("未缩放", scores_raw), ("除以 √d_k", scores_scaled)]:
    p = softmax(s)
    print(f"{name:>10}: scores = {s}, softmax = {p}, max = {p.max():.3f}, 熵 = {entropy(p):.3f}")

print("\n结论：把 key 乘以 2，夹角不变（余弦不变），但 attention 分数翻倍、分布更尖锐。")
print(f"余弦(q,k) = {cosine(q, k):.3f}，余弦(q,2k) = {cosine(q, 2*k):.3f}")


# ---------------------------------------------------------------- 5
section("5. 最小二乘 = 向列空间投影")
# 拟合 y = a + b x，数据点 (0,0), (1,1), (2,3)
xs = np.array([0.0, 1.0, 2.0])
ys = np.array([0.0, 1.0, 3.0])
A = np.column_stack([np.ones_like(xs), xs])  # [3, 2]
coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
pred = A @ coef
resid = ys - pred

print(f"A =\n{A}")
print(f"最小二乘系数 [截距, 斜率] = {coef}（解析解为 [-1/6, 3/2]）")
print(f"预测值 = {pred}")
print(f"残差 = {resid}")
print(f"Aᵀ·残差 = {A.T @ resid}（应接近 0，说明残差与列空间正交）")
print(f"残差范数 = {np.linalg.norm(resid):.3f}")

# 用正规方程手动求解，验证与 lstsq 一致
coef_normal = np.linalg.solve(A.T @ A, A.T @ ys)
print(f"正规方程解 = {coef_normal}")


# ---------------------------------------------------------------- 6
section("6. 小结")
print("1. 点积 = 长度 × 投影长度；余弦 = 去掉长度后的点积。")
print("2. 归一化使余弦 = 内积，是向量检索的常用优化。")
print("3. 投影的残差与子空间正交，这保证了“最近”。")
print("4. attention 用未归一化点积，因此对 key 的范数敏感。")

print("\n思考题：")
print("- 若把第 2 节所有向量都加上常数 1，余弦矩阵会怎样变化？动手改代码试试。")
print("- 若投影方向 u 不是单位向量，公式需要改吗？为什么？")
