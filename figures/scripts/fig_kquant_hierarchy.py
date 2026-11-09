#!/usr/bin/env python3
"""超级块层级图（纯标准库，无需 numpy/matplotlib）。

对应章节：13.7 块与超级块量化 §2.3
产物：figures/out/kquant-hierarchy.svg
运行：python3 figures/scripts/fig_kquant_hierarchy.py
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "out" / "kquant-hierarchy.svg"

svg_parts = []
svg_parts.append('<svg xmlns="http://www.w3.org/2000/svg" width="800" height="320" viewBox="0 0 800 320" font-family="sans-serif">')
svg_parts.append('<rect width="800" height="320" fill="#fffdf8"/>')
# titles
svg_parts.append('<text x="20" y="28" font-size="16" font-weight="bold" fill="#1f2933">K-quants 超级块的层次（256 值）</text>')
svg_parts.append('<text x="20" y="50" font-size="12" fill="#52606d">全局尺度 d、dmin + 8 个子块各自的 s_j、m_j；qs 为 4-bit 码</text>')
# outer superblock
svg_parts.append('<rect x="20" y="65" width="760" height="190" rx="8" fill="#eef2ff" stroke="#4f46e5" stroke-width="2"/>')
svg_parts.append('<text x="30" y="85" font-size="13" fill="#1f2933">超级块 256 值 · 144 字节（Q4_K）</text>')
svg_parts.append('<text x="30" y="105" font-size="12" fill="#52606d">d (fp16) + dmin (fp16) + scales[12]（8 对 6-bit）+ qs[128]</text>')
# 8 subblocks
for i in range(8):
    x = 30 + i * 92
    fill = "#fffdf8" if i % 2 == 0 else "#e6efff"
    svg_parts.append(f'<rect x="{x}" y="120" width="82" height="110" rx="4" fill="{fill}" stroke="#7b8794" stroke-width="1"/>')
    svg_parts.append(f'<text x="{x+41}" y="140" font-size="12" text-anchor="middle" fill="#1f2933">子块 {i}</text>')
    svg_parts.append(f'<text x="{x+41}" y="160" font-size="11" text-anchor="middle" fill="#52606d">32 值</text>')
    svg_parts.append(f'<text x="{x+41}" y="180" font-size="11" text-anchor="middle" fill="#b91c1c">s{i} · m{i}</text>')
    svg_parts.append(f'<text x="{x+41}" y="200" font-size="11" text-anchor="middle" fill="#52606d">q[0..31]</text>')
    svg_parts.append(f'<text x="{x+41}" y="218" font-size="10" text-anchor="middle" fill="#7b8794">6-bit</text>')
svg_parts.append('<text x="20" y="275" font-size="12" fill="#1f2933">反量化：y = d · s_j · q − dmin · m_j（解码直接乘整数，/63 已折进 d）</text>')
svg_parts.append('<text x="20" y="295" font-size="12" fill="#52606d">qs 交错：低半字节归偶数子块、高半字节归奇数子块（每 32 字节一轮）</text>')
svg_parts.append('</svg>')

OUT.write_text("\n".join(svg_parts), encoding="utf-8")
print("wrote", OUT.name)
