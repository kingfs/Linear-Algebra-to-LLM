import numpy as np
import matplotlib.pyplot as plt
import _style as st

rng = np.random.default_rng(7)
n = 24
x = np.linspace(0, 1, n)
u1 = np.sin(np.pi * x)
v1 = np.cos(0.5 * np.pi * x)
u2 = np.sin(3 * np.pi * x)
v2 = np.sin(2 * np.pi * x)
u3 = np.cos(5 * np.pi * x)
v3 = np.cos(3 * np.pi * x)
A = 4.0 * np.outer(u1, v1) + 1.2 * np.outer(u2, v2) + 0.5 * np.outer(u3, v3)
A = A + 0.06 * rng.normal(size=(n, n))

U, s, Vt = np.linalg.svd(A, full_matrices=False)
errors = []
recons = []
for k in [1, 2, 4]:
    Ak = (U[:, :k] * s[:k]) @ Vt[:k]
    recons.append(Ak)
    errors.append(np.linalg.norm(A - Ak) / np.linalg.norm(A))

fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.1))
vmax = np.abs(A).max()
images = [A] + recons
labels = ["A (24 x 24)", "rank 1", "rank 2", "rank 4"]
for i, (ax, img, label) in enumerate(zip(axes, images, labels)):
    ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if i == 0:
        ax.set_title(label, loc="left", fontsize=10)
    else:
        ax.set_title(f"{label}  (err {errors[i-1]:.0%})", loc="left", fontsize=10)

fig.suptitle("Low-rank approximation: keep the loudest stripes", y=1.03)
fig.tight_layout()
st.save(fig, "low-rank")
