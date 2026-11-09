#!/usr/bin/env python3
"""bpw 家族带图（纯标准库，无需 numpy/matplotlib）。

对应章节：13.8 码本与三值 §2 / §4
产物：figures/out/bpw-bands.svg
运行：python3 figures/scripts/fig_bpw_bands.py
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "out" / "bpw-bands.svg"

# (label, bpw, family)
data = [
    ("Q8_0", 8.5, "affine"),
    ("Q6_K", 6.5625, "kquant"),
    ("Q4_0", 4.5, "affine"),
    ("Q4_K", 4.5, "kquant"),
    ("IQ3_XXS", 3.0625, "codebook"),
    ("TQ2_0", 2.0625, "ternary"),
    ("TQ1_0", 1.6875, "ternary"),
    ("IQ1_S", 1.5625, "codebook"),
]
colors = {"affine": "#7b8794", "kquant": "#4f46e5", "codebook": "#0f766e", "ternary": "#d97706"}

W, H = 800, 360
left, right = 110, 760
xmax = 9.0
row_h = 28
top = 55

parts = []
parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="sans-serif">')
parts.append(f'<rect width="{W}" height="{H}" fill="#fffdf8"/>')
parts.append('<text x="20" y="28" font-size="16" font-weight="bold" fill="#1f2933">每权重比特（bpw）家族带</text>')
parts.append('<text x="20" y="48" font-size="12" fill="#52606d">块字节数 × 8 ÷ 每块权重数；已含尺度/索引的摊销</text>')
# x axis grid
for v in [0, 2, 4, 6, 8]:
    x = left + (right - left) * (v / xmax)
    parts.append(f'<line x1="{x}" y1="{top-5}" x2="{x}" y2="{top+len(data)*row_h+5}" stroke="#e6e0d4" stroke-width="1"/>')
    parts.append(f'<text x="{x}" y="{top+len(data)*row_h+22}" font-size="11" text-anchor="middle" fill="#52606d">{v}</text>')
for i, (label, bpw, fam) in enumerate(data):
    y = top + i * row_h
    w = (right - left) * (bpw / xmax)
    parts.append(f'<text x="10" y="{y+15}" font-size="12" fill="#1f2933">{label}</text>')
    parts.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="18" rx="3" fill="{colors[fam]}"/>')
    parts.append(f'<text x="{left+w+6:.1f}" y="{y+14}" font-size="11" fill="#1f2933">{bpw:g}</text>')
legend_y = top + len(data) * row_h + 45
for j, (k, c) in enumerate(colors.items()):
    x = 20 + j * 150
    parts.append(f'<rect x="{x}" y="{legend_y-12}" width="14" height="14" fill="{c}"/>')
    parts.append(f'<text x="{x+20}" y="{legend_y}" font-size="12" fill="#1f2933">{k}</text>')
parts.append('</svg>')

OUT.write_text("\n".join(parts), encoding="utf-8")
print("wrote", OUT.name)
