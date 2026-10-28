import numpy as np
import matplotlib.pyplot as plt
import _style as st

rng = np.random.default_rng(11)
n = 60
block = 12
A = np.zeros((n, n))
for b in range(5):
    i0 = b * block
    for i in range(i0, i0 + block):
        A[i, i] = 1.0
        for j in range(i0, i0 + block):
            if rng.random() < 0.28:
                A[i, j] = 1.0
for b in range(4):
    i0 = b * block
    A[i0:i0 + block, i0 + block:i0 + 2 * block] = (rng.random((block, block)) < 0.06)

nnz = int(A.sum())
B = np.zeros(n * n, dtype=bool)
B[rng.choice(n * n, size=nnz, replace=False)] = True
B = B.reshape(n, n)

fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))
ax = axes[0]
ax.spy(A, markersize=2.2, color=st.ACCENT)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Structured: blocks and bands", loc="left", fontsize=11)
ax.text(0.02, -0.10, f"nnz={nnz}, density={nnz/n**2:.1%}", transform=ax.transAxes,
        fontsize=9, color=st.GRAY)

ax = axes[1]
ax.spy(B, markersize=2.2, color=st.GRAY)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Unstructured: random nonzeros", loc="left", fontsize=11)
ax.text(0.02, -0.10, f"nnz={nnz}, density={nnz/n**2:.1%}", transform=ax.transAxes,
        fontsize=9, color=st.GRAY)

fig.suptitle("Same number of nonzeros, very different hardware friendliness", y=1.02)
fig.tight_layout()
st.save(fig, "sparse-pattern")
