"""M14 figure: benchmark families and their version timelines (2020–2026).

示意：横轴按首次发布/主要版本的大致时间放置，不追求精确到日。
"""

import matplotlib.pyplot as plt

from _style import ACCENT, ORANGE, GREEN, RED, GRAY, save

CATS = {
    "Knowledge": ACCENT,
    "Math": "#7c3aed",
    "Code": GREEN,
    "Agent": ORANGE,
    "Instruction / preference": RED,
    "Multimodal": "#0369a1",
    "Long context": GRAY,
    "Safety": "#4d7c0f",
}

# (family, [(x, label), ...]); label 为空表示只画点不写字。
ROWS = [
    ("Knowledge", [
        ("MMLU", [(2020.75, ""), (2024.5, "Pro / Redux")]),
        ("GPQA", [(2023.9, ""), (2024.2, "Diamond")]),
        ("HLE", [(2025.0, ""), (2025.5, "text / tools")]),
        ("SimpleQA", [(2024.9, "")]),
        ("AA-Omniscience", [(2025.9, "")]),
    ]),
    ("Math", [
        ("AIME", [(2024.1, "2024"), (2025.1, "2025"), (2026.1, "2026")]),
        ("MATH / GSM8K", [(2021.5, "")]),
    ]),
    ("Code", [
        ("HumanEval / MBPP", [(2021.5, "")]),
        ("LiveCodeBench", [(2024.2, ""), (2025.2, ""), (2025.45, "v6 / Pro")]),
        ("SWE-bench", [(2023.8, ""), (2024.6, ""), (2024.8, "Verified / MM"),
                       (2025.3, ""), (2025.7, "Pro"), (2026.8, "Pro V2")]),
        ("Terminal-Bench", [(2025.4, ""), (2025.9, "2.0"), (2026.4, "2.1 → 4.0"),
                            (2026.6, ""), (2026.7, "")]),
        ("NL2Repo-Bench", [(2025.95, "")]),
        ("DeepSWE / ProgramBench", [(2026.4, ""), (2026.5, "")]),
    ]),
    ("Agent", [
        ("BFCL", [(2024.1, ""), (2024.6, ""), (2024.7, "v2 / v3"), (2025.95, "v4")]),
        ("τ-bench", [(2024.5, ""), (2025.5, "τ²"), (2026.2, "τ³")]),
        ("WebArena", [(2023.5, ""), (2025.5, "Verified")]),
        ("OSWorld", [(2024.3, ""), (2025.5, "Verified")]),
        ("MCP-Atlas", [(2026.0, "")]),
        ("Agents' Last Exam", [(2026.5, "")]),
    ]),
    ("Instruction / preference", [
        ("IFEval", [(2023.9, "")]),
        ("IFBench", [(2025.4, "")]),
        ("MT-Bench", [(2023.5, "")]),
        ("AlpacaEval / Arena-Hard", [(2024.3, ""), (2024.5, "")]),
        ("LMArena", [(2023.4, "")]),
    ]),
    ("Multimodal", [
        ("MMMU", [(2023.9, ""), (2024.7, "Pro")]),
        ("MathVista / MathVision", [(2023.8, ""), (2024.1, "MathVision")]),
        ("CharXiv / Chartography", [(2024.5, ""), (2026.6, "")]),
        ("OmniDocBench", [(2024.95, ""), (2026.5, "v1.5")]),
        ("BabyVision", [(2026.0, "")]),
    ]),
    ("Long context", [
        ("RULER", [(2024.3, "")]),
        ("LongBench v2", [(2024.95, "")]),
        ("AA-LCR", [(2025.6, "")]),
    ]),
    ("Safety", [
        ("XSTest", [(2023.62, "")]),
        ("StrongREJECT", [(2024.15, "")]),
        ("HarmBench", [(2024.28, "")]),
        ("FORTRESS", [(2025.45, "")]),
    ]),
]

fig, ax = plt.subplots(figsize=(11.2, 11.4))

y = 0.0
for cat, rows in ROWS:
    ax.text(2019.0, y + 0.15, cat, ha="left", va="center",
            fontsize=9.5, fontweight="bold", color=CATS[cat])
    y -= 1.15
    for name, marks in rows:
        ax.text(2019.0, y, name, ha="left", va="center", fontsize=7.8, color="#1f2933")
        ax.plot([2019.55, 2026.85], [y, y], color=CATS[cat], lw=5, alpha=0.16,
                solid_capstyle="round", zorder=1)
        for i, (x, label) in enumerate(marks):
            ax.plot([x], [y], "o", ms=5.2, color=CATS[cat], zorder=3,
                    markeredgecolor="#fffdf8", markeredgewidth=0.8)
            if label:
                dy = 7 if i % 2 == 0 else -11
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, dy),
                            ha="center", fontsize=6.4, color=CATS[cat], zorder=4)
        y -= 1.0
    y -= 0.65

ax.set_xlim(2018.85, 2027.1)
ax.set_ylim(y + 0.55, 1.45)
ax.set_xticks(range(2020, 2027))
ax.set_xticklabels([str(t) for t in range(2020, 2027)], fontsize=9)
ax.set_yticks([])
for spine in ("left", "right", "top"):
    ax.spines[spine].set_visible(False)
ax.grid(axis="x", color="#e6e0d4", lw=0.8, zorder=0)
ax.set_axisbelow(True)
ax.set_title("Benchmark families, 2020–2026: dot = major release",
             fontsize=13, loc="left", pad=12)
ax.text(2018.9, 0.95, "bar = active span; positions are approximate first-release dates; "
        "version labels are illustrative, not exhaustive",
        fontsize=7.6, color=GRAY, ha="left")

save(fig, "m14-benchmark-timeline")
