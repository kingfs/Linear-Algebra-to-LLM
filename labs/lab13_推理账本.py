"""Lab13 · 推理账本：显存、带宽与算术强度

对应章节：13.1 推理的数学 / 13.2 模型格式与量化 / 11.7 KV Cache

验证：
1. KV Cache 的字节公式，以及 GQA、混合架构如何改变它；
2. prefill 与 decode 的算术强度，以及 Roofline 拐点在哪；
3. 用「每 token 读取字节 ÷ 带宽」估算 TPOT；
4. 量化的端到端加速上界，为什么短上下文与长上下文差一个数量级；
5. MoE 只读活跃专家时，tokens/s 的上限由什么决定。

运行：uv run --with numpy python labs/lab13_推理账本.py
"""

import numpy as np

GIB = 2 ** 30          # 1 GiB = 2^30 B
GB = 1e9               # 十进制 GB


def section(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def check(name: str, ok: bool) -> None:
    print(f"[{'通过' if ok else '失败'}] {name}")


def kv_bytes(layers: int, kv_heads: int, head_dim: int, tokens: int,
             bytes_per: int = 2) -> int:
    """标准 MHA/GQA 的 KV 字节数：2(K,V) · L · H_kv · d_k · T · b。"""
    return 2 * layers * kv_heads * head_dim * tokens * bytes_per


# ---------------------------------------------------------------- 1
section("1. KV Cache：同一句「7B / 32k」，MHA 与 GQA 差 4 倍")
L, dk, T, b = 32, 128, 32768, 2
for hkv in (32, 8):
    n = kv_bytes(L, hkv, dk, T, b)
    print(f"  H_kv={hkv:2d}: {n / 1024:,.0f} KiB/token … 32k 合计 {n / GIB:6.2f} GiB")
check("H_kv=32 时 32k KV = 16 GiB", abs(kv_bytes(L, 32, dk, T, b) / GIB - 16) < 0.01)
check("H_kv=8 时 32k KV = 4 GiB", abs(kv_bytes(L, 8, dk, T, b) / GIB - 4) < 0.01)

# 混合架构：只有 global 层产生全量 KV，SWA 层被窗口截断
L_g, H_g, T128 = 6, 4, 131072
n_global = kv_bytes(L_g, H_g, dk, T128, b)
n_swa = kv_bytes(5, H_g, dk, 256, b)          # 5 层 SWA，窗口 256
print(f"\n  LFM3 式（6 global + 5 SWA，H_kv=4）：")
print(f"    global {n_global / GIB:6.2f} GiB + SWA {n_swa / 2**20:6.2f} MiB "
      f"= {(n_global + n_swa) / GIB:6.2f} GiB @128k")
check("LFM3 式 128k KV ≈ 1.5 GiB", abs((n_global + n_swa) / GIB - 1.5) < 0.05)


# ---------------------------------------------------------------- 2
section("2. 算术强度：prefill 约 T，decode 约 2/b")
def ai_prefill(T_tokens: int, bytes_per: int = 2) -> float:
    """2·N·T FLOP / (N·b 字节) = 2T/b。"""
    return 2 * T_tokens / bytes_per


def ai_decode(bytes_per: int) -> float:
    """2·N FLOP / (N·b 字节) = 2/b。"""
    return 2 / bytes_per


for t in (128, 512, 1024, 4096):
    print(f"  prefill T={t:5d}: 算术强度 {ai_prefill(t):8.1f} FLOP/B")
for name, bp in (("fp32", 4), ("fp16", 2), ("int4", 0.5)):
    print(f"  decode {name:4s}: 算术强度 {ai_decode(bp):6.2f} FLOP/B")
check("prefill T=1024 约 1024 FLOP/B", abs(ai_prefill(1024) - 1024) < 1e-9)
check("decode fp16 = 1 FLOP/B", abs(ai_decode(2) - 1) < 1e-12)
check("decode int4 = 4 FLOP/B", abs(ai_decode(0.5) - 4) < 1e-12)

# Roofline 拐点：峰值算力 ÷ 带宽（H100 级，FP16 稠密）
peak_flops, bw = 989e12, 3.35e12
ridge = peak_flops / bw
print(f"\n  拐点 = {peak_flops / 1e12:.0f} TFLOP/s ÷ {bw / 1e12:.2f} TB/s "
      f"= {ridge:.0f} FLOP/B")
print(f"  → decode(1 FLOP/B) 在拐点左侧 {ridge:.0f}×；prefill 需 T ≳ {ridge * 2 / 2:.0f} 才越过")
check("decode 距拐点约两个数量级",
      1 < ridge / ai_decode(2) < 1000 and ridge / ai_decode(2) > 100)


# ---------------------------------------------------------------- 3
section("3. TPOT：每 token 读多少字节，就等多久")
W_gib = 14e9 / GIB                       # 7B fp16 = 14 GB ≈ 13.04 GiB
HBM = 2000.0                             # GiB/s（按 2 TB/s 估）
for ctx in (0, 32768, 131072):
    kv = kv_bytes(L, 8, dk, ctx, b) / GIB
    total = W_gib + kv
    print(f"  ctx={ctx:6d}: 读 {W_gib:5.2f}+{kv:5.2f}={total:6.2f} GiB "
          f"→ TPOT {total / HBM * 1000:5.1f} ms，单请求 {HBM / total:5.1f} token/s")
check("无 KV 时 TPOT ≈ 7 ms", abs(W_gib / HBM * 1000 - 6.5) < 1.5)


# ---------------------------------------------------------------- 4
section("4. 量化收益上界：同一「2×压缩」，4k 值 7%、128k 值 71%")
def speedup_bound(w_gib: float, kv_gib: float, q_w: float = 1.0,
                  q_kv: float = 2.0) -> float:
    return (w_gib + kv_gib) / (w_gib / q_w + kv_gib / q_kv)


for ctx, kv_gib in ((4096, 2.0), (131072, 64.0)):
    s = speedup_bound(W_gib, kv_gib)
    print(f"  ctx={ctx:6d}, W={W_gib:.2f} GiB, KV={kv_gib:5.1f} GiB → 上界 {s:.3f}×")
    check(f"ctx={ctx} 的上界 ≈ {1.067 if ctx == 4096 else 1.71:.2f}",
          abs(s - (1.067 if ctx == 4096 else 1.71)) < 0.02)


# ---------------------------------------------------------------- 5
section("5. MoE：活跃参数决定每 token 读多少")
total_b, active_b, bits = 100e9, 5e9, 4
bytes_per_param = bits / 8
active_bytes = active_b * bytes_per_param
for io_bw in (5.0, 10.0, 50.0):          # GB/s：SSD / PCIe / 内存
    print(f"  活跃专家 {active_bytes / GB:.1f} GB，带宽 {io_bw:4.1f} GB/s "
          f"→ 上限 {io_bw / (active_bytes / GB):5.2f} token/s")
check("5 GB/s SSD 下约 2 token/s",
      abs(5.0 / (active_bytes / GB) - 2.0) < 0.1)
print(f"\n  总参数 {total_b / 1e9:.0f}B × 0.5 B = {total_b * bytes_per_param / GB:.0f} GB："
      "决定「放在哪」，不决定「每 token 读多少」。")


# ---------------------------------------------------------------- 6
section("6. 总结")
print("1) KV = 2·L·H_kv·d_k·T·b；GQA 改 H_kv，混合架构改「哪些层算 KV」。")
print("2) 算术强度：prefill ≈ 2T/b，decode ≈ 2/b；拐点 = 算力/带宽。")
print("3) TPOT ≈ (权重 + KV) / 带宽；长上下文先被 KV 卡住。")
print("4) 量化上界 = (W+KV)/(W/q_w + KV/q_kv)；把压缩率当加速比是空头支票。")
print("5) MoE 的 tokens/s 由「活跃字节 ÷ 可用带宽」决定，而不是总参数量。")
print("\n一句话：显存决定「放不放得下」，带宽决定「快不快」，算力只在 prefill 站出来。")
