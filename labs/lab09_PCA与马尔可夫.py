"""Lab09 · PCA 与马尔可夫链

对应章节：9.1 概率与随机变量 / 9.2 多元高斯 / 9.3 PCA / 9.4 白化 / 9.5 马尔可夫链

验证六件事：
1. 样本协方差矩阵与特征分解，特征向量就是主方向；
2. PCA 的投影、重建，以及重建误差 = 丢弃特征值之和 × (n-1)；
3. 解释方差比 = 奇异值平方 / 总奇异值平方；
4. PCA 白化与 ZCA 白化都把协方差变成单位阵，而标准化只得到相关系数矩阵；
5. 马尔可夫链的行随机矩阵幂迭代收敛到稳态，速度由第二特征值控制；
6. 二维高斯的等高线是椭圆，半轴为 sqrt(特征值)，马氏距离平方服从卡方分布。

运行：uv run --with numpy python labs/lab09_PCA与马尔可夫.py
依赖：numpy（不依赖 matplotlib）
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)
RNG = np.random.default_rng(0)


def section(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


# ================================================================ 1
section("1. 样本协方差与特征分解")

TRUE_COV = np.array([[2.0, 1.0],
                     [1.0, 2.0]])
L = np.linalg.cholesky(TRUE_COV)
n = 400
X = RNG.normal(size=(n, 2)) @ L.T          # [n, 2]，真实协方差 = TRUE_COV

mu = X.mean(axis=0)
Xc = X - mu
S = Xc.T @ Xc / (n - 1)                    # [2, 2] 无偏样本协方差

lam, V = np.linalg.eigh(S)                 # 升序
order = np.argsort(lam)[::-1]              # 改成降序
lam, V = lam[order], V[:, order]

print(f"样本均值 mu = {mu}")
print(f"样本协方差 S =\n{S}")
print(f"真实协方差 Sigma =\n{TRUE_COV}")
print(f"特征值（降序） = {lam}")
print(f"特征向量（列） =\n{V}")
print("结论：样本协方差接近真实值；特征向量给出点云的两个主轴。")

# ================================================================ 2
section("2. PCA 投影与重建")

# 与特征分解等价的 SVD
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
print(f"奇异值 s = {s}")
print(f"s^2 / (n-1) = {s**2 / (n - 1)}   （应等于上面的特征值）")

k = 1
Vk = V[:, :k]                              # [2, 1]
scores = Xc @ Vk                           # [n, 1]
Xhat = scores @ Vk.T + mu                  # [n, 2]
err = ((X - Xhat) ** 2).sum()              # 总平方重建误差

print(f"\n取 k = {k}")
print(f"前 5 个点的得分 = {scores[:5, 0]}")
print(f"前 5 个点的重建 = \n{Xhat[:5]}")
print(f"总平方重建误差 = {err:.6f}")
print(f"丢弃特征值之和 × (n-1) = {(n - 1) * lam[k:].sum():.6f}")
print("结论：两者相等；PCA 的误差完全由被丢掉的特征值决定。")

# ================================================================ 3
section("3. 解释方差比")

ratio = s**2 / (s**2).sum()
cum = np.cumsum(ratio)
print(f"解释方差比 = {ratio}")
print(f"累计解释方差比 = {cum}")
print(f"前 1 个主成分保留 {cum[0]:.2%} 的总方差")

# 高维例子：3 个潜在因子 + 噪声
D = 6
latent = RNG.normal(size=(2000, 3)) @ np.diag([3.0, 2.0, 1.0])
noise = 0.3 * RNG.normal(size=(2000, D))
Q, _ = np.linalg.qr(RNG.normal(size=(D, D)))   # 随机正交基
X6 = latent @ Q[:, :3].T + noise

X6c = X6 - X6.mean(axis=0)
_, s6, _ = np.linalg.svd(X6c, full_matrices=False)
r6 = s6**2 / (s6**2).sum()
print(f"\n6 维数据的前 3 个解释方差比 = {r6[:3]}")
print(f"前 3 个累计 = {r6[:3].sum():.2%}；前 5 个累计 = {r6[:5].sum():.2%}")
print("结论：谱衰减快时，少数主成分就能解释大部分方差。")

# ================================================================ 4
section("4. 白化：PCA / ZCA / 只标准化")

W_pca = np.diag(1 / np.sqrt(lam)) @ V.T        # [2, 2]
Z_pca = Xc @ W_pca.T
cov_pca = np.cov(Z_pca, rowvar=False)

W_zca = V @ np.diag(1 / np.sqrt(lam)) @ V.T    # 对称逆平方根
Z_zca = Xc @ W_zca.T
cov_zca = np.cov(Z_zca, rowvar=False)

X_std = Xc / Xc.std(axis=0, ddof=1)
cov_std = np.cov(X_std, rowvar=False)

print(f"PCA 白化矩阵 W_pca =\n{W_pca}")
print(f"ZCA 白化矩阵 W_zca =\n{W_zca}")
print(f"\nPCA 白化后的协方差 =\n{cov_pca}")
print(f"ZCA 白化后的协方差 =\n{cov_zca}")
print(f"只标准化后的协方差（相关系数矩阵）=\n{cov_std}")
print(f"\nPCA 白化偏离 I 的最大值 = {np.abs(cov_pca - np.eye(2)).max():.2e}")
print(f"ZCA 白化偏离 I 的最大值 = {np.abs(cov_zca - np.eye(2)).max():.2e}")
print("结论：两种白化都得到单位阵；标准化只把对角变成 1，非对角相关仍在。")

# ================================================================ 5
section("5. 马尔可夫链：幂迭代到稳态")

P = np.array([[0.9, 0.1],
              [0.2, 0.8]])       # 行随机：每行和为 1
pi = np.array([1.0, 0.0])        # 行向量，从状态 0 出发
pi_star = np.array([2 / 3, 1 / 3])

print(f"转移矩阵 P =\n{P}")
print(f"行和 = {P.sum(axis=1)}")
print(f"特征值 = {np.linalg.eigvals(P)}")
print(f"解析稳态 = {pi_star}")

print("\n迭代过程：")
for t in range(1, 31):
    pi = pi @ P
    if t in (1, 2, 3, 5, 10, 20, 30):
        gap = np.abs(pi - pi_star).max()
        print(f"  t={t:2d}: pi = {pi}，与稳态最大偏差 = {gap:.6f}")

print(f"\n解析公式 pi_1(t) = 2/3 + (1/3) * 0.7^t")
for t in (5, 10, 20):
    print(f"  t={t:2d}: {2/3 + (1/3) * 0.7**t:.6f}")

# 三状态 PageRank 玩具：A→B,C；B→C；C→A
M = np.array([[0.0, 0.5, 0.5],
              [0.0, 0.0, 1.0],
              [1.0, 0.0, 0.0]])
q = np.array([1.0, 0.0, 0.0])
for _ in range(100):
    q = q @ M
print(f"\n三状态玩具链的稳态 = {q}（解析值 [0.4, 0.2, 0.4]）")
print(f"特征值 = {np.linalg.eigvals(M)}")
print("结论：行随机矩阵的幂迭代收敛到 pi P = pi；收敛速度由 |lambda_2| 决定。")

# ================================================================ 6
section("6. 二维高斯：协方差决定椭圆")

Sigma = np.array([[2.0, 1.0],
                  [1.0, 2.0]])
lam_g, V_g = np.linalg.eigh(Sigma)
order_g = np.argsort(lam_g)[::-1]
lam_g, V_g = lam_g[order_g], V_g[:, order_g]
Sigma_inv = np.linalg.inv(Sigma)

print(f"Sigma =\n{Sigma}")
print(f"det(Sigma) = {np.linalg.det(Sigma):.4f}")
print(f"特征值 = {lam_g}，半轴（c=1）= {np.sqrt(lam_g)}")
print(f"特征向量（列）=\n{V_g}")

print("\n等高线采样（c=1 与 c=2），并核对马氏距离平方：")
for c in (1.0, 2.0):
    print(f"  c={c}:")
    for deg in (0, 45, 90, 135):
        theta = np.deg2rad(deg)
        y = c * np.sqrt(lam_g) * np.array([np.cos(theta), np.sin(theta)])
        x = V_g @ y
        q = x @ Sigma_inv @ x
        print(f"    theta={deg:3d}°: x=({x[0]: .4f}, {x[1]: .4f}), "
              f"马氏距离^2={q:.4f}（应={c*c:.4f}）")

# 用采样验证卡方分布：二维高斯落在 c 倍等高线内的概率 = 1 - exp(-c^2/2)
samples = RNG.multivariate_normal(np.zeros(2), Sigma, size=200000)
q_samples = np.einsum("ij,jk,ik->i", samples, Sigma_inv, samples)
for c in (1.0, 2.0):
    frac = np.mean(q_samples <= c * c)
    theory = 1 - np.exp(-c * c / 2)
    print(f"c={c}: 落在等高线内的比例 = {frac:.4f}，理论值 = {theory:.4f}")

print("\n结论：高斯的等高线是椭圆，半轴为 sqrt(lambda_i)，方向为特征向量。")

# ================================================================ 7
section("小结")
print("1) 样本协方差是数据的 Gram 矩阵，特征分解给出主轴与方差。")
print("2) PCA = 中心化 + SVD；重建误差 = 丢弃奇异值平方和。")
print("3) 解释方差比 = s_i^2 / sum(s_j^2)。")
print("4) 白化把协方差变成 I；标准化只得到相关系数矩阵。")
print("5) 马尔可夫链的幂迭代收敛到 pi P = pi，速度由 |lambda_2| 控制。")
print("6) 二维高斯的椭圆半轴是 sqrt(特征值)，不是特征值本身。")

# 思考题：
# 1. 若 n < D，第 1 节的 S 会有什么问题？特征值有几个非零？
# 2. 把第 5 节的 P 改成 [[0,1],[1,0]]，幂迭代还收敛吗？为什么？
# 3. 若把第 4 节的标准化换成 ZCA，白化后的点与原点的距离会怎样变化？
