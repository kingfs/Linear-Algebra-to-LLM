"""Lab14 · GGUF 块量化：Q4_0 与 Q4_K 的字节级解码

对应章节：11.9 量化 / 13.2 模型格式与量化 / 13.7 块与超级块量化 / 13.8 码本与三值

验证：
1. Q4_0（18 字节/32 值）与 Q4_K（144 字节/256 值）的反量化公式与打包位序；
2. K-quants 的 6-bit 归一化：解码端只乘整数，/63 已折进 d/dmin；
3. 块内异质性下子块修正为什么能减小误差；
4. 各格式的块级 bpw 自算；
5. imatrix 加权误差与三值 base-3 打包。

运行：python3 labs/lab14_gguf量化.py
（仅标准库，无需 numpy。）
"""

import math
import random
import struct


def section(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def check(name: str, ok: bool) -> None:
    print(f"[{'通过' if ok else '失败'}] {name}")


def f16_to_f32(b2: bytes) -> float:
    return struct.unpack("<e", b2)[0]


def f32_to_f16(x: float) -> bytes:
    return struct.pack("<e", x)


# ---------------------------------------------------------------- 1
section("1. Q4_0：18 字节 / 32 值，y = d*(q-8)")
# 构造一块：d=0.1，qs[0]=0x5C（低=12，高=5），其余 nibble = 8（即 0）
block = bytearray(18)
block[0:2] = f32_to_f16(0.1)
block[2] = 0x5C
for j in range(1, 16):
    block[2 + j] = 0x88  # nibble 8 → (8-8)=0


def deq_q4_0(b: bytes):
    d = f16_to_f32(b[:2])
    y = [0.0] * 32
    for j in range(16):
        y[j] = d * ((b[2 + j] & 0xF) - 8)
        y[j + 16] = d * ((b[2 + j] >> 4) - 8)
    return d, y


d, y = deq_q4_0(bytes(block))
print(f"  d={d!r}（fp16 存 0.1 的往返值）")
print(f"  y[0]={y[0]:.6f}（期望 约 0.4），y[16]={y[16]:.6f}（期望 约 -0.3）")
check("Q4_0 y[0] ≈ 0.4（fp16 底噪）", abs(y[0] - 0.4) < 1e-3)
check("Q4_0 y[16] ≈ -0.3（fp16 底噪）", abs(y[16] - (-0.3)) < 1e-3)
rest = sum(abs(v) for v in y[1:16]) + sum(abs(v) for v in y[17:])
check("Q4_0 其余为 0", rest < 1e-9)
check("Q4_0 bpw = 4.5", 18 * 8 / 32 == 4.5)


# ---------------------------------------------------------------- 2
section("2. Q4_K：144 字节 / 256 值，y = d*sc*q - dmin*m（只乘不除）")
# get_scale_min_k4 与 llama.cpp ggml-quants.c 逐行对应
def get_scale_min_k4(j: int, scales: bytes):
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
    m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return d, m


def deq_q4_K(b: bytes):
    assert len(b) == 144
    d = f16_to_f32(b[0:2])
    dmin = f16_to_f32(b[2:4])
    scales = b[4:16]
    qs = b[16:144]
    y = []
    is_ = 0
    qpos = 0
    for _ in range(4):  # 4 × 64 = 256
        sc1, m1 = get_scale_min_k4(is_, scales)
        sc2, m2 = get_scale_min_k4(is_ + 1, scales)
        d1, mm1 = d * sc1, dmin * m1
        d2, mm2 = d * sc2, dmin * m2
        for l in range(32):
            y.append(d1 * (qs[qpos + l] & 0xF) - mm1)
        for l in range(32):
            y.append(d2 * (qs[qpos + l] >> 4) - mm2)
        qpos += 32
        is_ += 2
    return d, dmin, y


# 构造最小块：d=0.02, dmin=0.01, 子块0 的 (sc,m)=(10,5)
# scales[0]=10（j=0 的 d），scales[4]=5（j=0 的 m），qs[0]=0x07（q=7）
kb = bytearray(144)
kb[0:2] = f32_to_f16(0.02)
kb[2:4] = f32_to_f16(0.01)
kb[4] = 10
kb[8] = 5
kb[16] = 0x07  # qs[0]
d_, dmin_, yk = deq_q4_K(bytes(kb))
print(f"  d={d_!r} dmin={dmin_!r}")
print(f"  y[0]={yk[0]:.6f}（期望 约 1.35），y[32]={yk[32]:.6f}（子块1 首值）")
check("Q4_K y[0] = d*10*7 - dmin*5 ≈ 1.35", abs(yk[0] - 1.35) < 1e-3)
check("Q4_K 长度 256", len(yk) == 256)
check("Q4_K bpw = 4.5", 144 * 8 / 256 == 4.5)
print(f"  若解码端误除以 63：y[0] 会变成 {yk[0] / 63:.6f}（缩 63 倍，logits 垮掉）")
check("误除 63 会缩 63 倍", abs(yk[0] / 63 - 1.35 / 63) < 1e-4)


# ---------------------------------------------------------------- 3
section("3. 异质性实验：一个大尺子 vs 八把子尺子")
random.seed(0)
# 零均值但尺度不同：前半小尺度、后半大尺度（对称量化能表达的异质性）
x = [random.gauss(0, 0.01) for _ in range(128)] + \
    [random.gauss(0, 0.5) for _ in range(128)]


def quant_sym(v):
    s = (max(abs(u) for u in v) / 7.0) if v else 0.0
    if s == 0:
        return [0.0] * len(v)
    out = []
    for u in v:
        q = round(u / s)
        q = max(-8, min(7, q))
        out.append(q * s)
    return out


def rmse(a, b):
    return math.sqrt(sum((u - v) ** 2 for u, v in zip(a, b)) / len(a))


y_single = quant_sym(x)
y_sub = []
for j in range(8):
    y_sub.extend(quant_sym(x[j * 32:(j + 1) * 32]))
rmse_s, rmse_sub = rmse(x, y_single), rmse(x, y_sub)
# 小尺度半块的 RMSE 才是子块修正的痛点：单尺度被大尺度撑粗
rmse_s_small = rmse(x[:128], y_single[:128])
rmse_sub_small = rmse(x[:128], y_sub[:128])
print(f"  全局单尺度 RMSE = {rmse_s:.4f}；子块尺度 RMSE = {rmse_sub:.4f}")
print(f"  仅小尺度半块：单尺度 {rmse_s_small:.4f} vs 子块 {rmse_sub_small:.4f}")
check("整体子块更小", rmse_sub < rmse_s)
check("小尺度半块改善显著", rmse_sub_small < rmse_s_small * 0.5)
# 均匀分布时差距收窄
x2 = [random.gauss(0, 0.3) for _ in range(256)]
y2s = quant_sym(x2)
y2sub = []
for j in range(8):
    y2sub.extend(quant_sym(x2[j * 32:(j + 1) * 32]))
print(f"  均匀分布时：单尺度 {rmse(x2, y2s):.4f} vs 子块 {rmse(x2, y2sub):.4f}（差距收窄）")
check("均匀分布时差距收窄", rmse(x2, y2sub) < rmse(x2, y2s) * 0.9)


# ---------------------------------------------------------------- 4
section("4. 块级 bpw 家族地图（块字节×8÷每块权重数）")
table = [("Q4_0", 18, 32), ("Q8_0", 34, 32), ("Q2_K", 84, 256),
         ("Q3_K", 110, 256), ("Q4_K", 144, 256), ("Q5_K", 176, 256),
         ("Q6_K", 210, 256)]
for name, nbytes, nval in table:
    print(f"  {name}: {nbytes * 8 / nval:.4f} bpw")
check("Q5_K = 5.5", 176 * 8 / 256 == 5.5)
check("Q6_K ≈ 6.5625", abs(210 * 8 / 256 - 6.5625) < 1e-9)


# ---------------------------------------------------------------- 5
section("5. 总结")
print("1) Q4_0: y=d*(q-8)，低半字节是前 16 个、高半字节是后 16 个。")
print("2) Q4_K: y=d*sc*q-dmin*m，6-bit 只乘不除（/63 已折进 d）。")
print("3) 子块修正治的是尺度失配：异质块里单尺子误差大一个量级。")
print("\n思考题：把 y[0] 的 nibble 改成 15，结果变成多少？这对应为什么误差？")


# ---------------------------------------------------------------- 6
section("6. 加权误差与三值打包（13.8）")
# 加权 MSE：激活越大的方向误差越贵
def weighted_rmse(a, b, alphas):
    return math.sqrt(sum(al * (x - y) ** 2 for x, y, al in zip(a, b, alphas)) / sum(alphas))

w = [2.0, 0.1, -0.05, 0.02]
alphas = [4.0, 0.5, 0.5, 0.5]
w_hat = [1.9, 0.0, 0.0, 0.0]
print(f"  加权 RMSE = {weighted_rmse(w, w_hat, alphas):.4f}")
print(f"  均匀 RMSE = {math.sqrt(sum((x - y) ** 2 for x, y in zip(w, w_hat)) / len(w)):.4f}")
check("加权误差可算", weighted_rmse(w, w_hat, alphas) > 0)
# 三值 base-3：5 trit 打包进 1 字节（3^5=243<256）
trits = [2, 0, 1, 1, 0]  # 2=+d, 0=0, 1=-d 的一种映射
n_val = sum(t * (3 ** (4 - i)) for i, t in enumerate(trits))
print(f"  trit {trits} -> N={n_val}（<256，打包 1 字节）")
check("5 trit 打包", n_val == 174 and n_val < 256)
# 阈值效应示意：同一加权误差下，推理任务对持续偏差更敏感（这里只算数字，不做任务评测）
print(f"  log2(3)={math.log2(3):.4f} bit/值（理论下限）；5 trit/字节={8/5:.3f} bpw（真实，未含 d）")
check("log2(3)≈1.585", abs(math.log2(3) - 1.585) < 0.001)
