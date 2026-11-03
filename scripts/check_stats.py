#!/usr/bin/env python3
"""README 规模统计与仓库实际数量的一致性检查。

用法：
    python3 scripts/check_stats.py
    python3 scripts/check_stats.py --quiet

背景：README 首页会写「N 个模块 · N 篇概念文档 · N 组习题 · N 个 lab ·
N 道可判分题 · N 张图」。这些数字一旦手写就会漂移（曾出现「93 篇 / 406 题 /
62 张图」落后于实际的「100 篇 / 434 题 / 64 张图」）。本脚本把它们钉在文件系统上。

检查项（只校验能找到对应文案的数字，找不到就跳过，避免措辞变化导致误报）：
1. `N 个模块`、`N 篇概念`、`N 组习题`、`N 个可运行 lab` / `N 个 lab`、
   `N 道`（可判分题）、`N 张图` 与磁盘上的实际数量一致。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
MODULE_DIR_RE = re.compile(r"^\d{2}-.+$")
CONCEPT_FILE_RE = re.compile(r"^\d{2}-.+\.md$")


def actual() -> dict[str, int]:
    modules = [d for d in ROOT.iterdir() if d.is_dir() and MODULE_DIR_RE.match(d.name)]
    concepts = [
        f
        for d in modules
        for f in d.glob("*.md")
        if CONCEPT_FILE_RE.match(f.name)
    ]
    labs = list((ROOT / "labs").glob("lab*.py"))
    exercises = list((ROOT / "exercises").glob("m*-习题.md"))
    figures = list((ROOT / "figures" / "out").glob("*.svg"))
    questions = 0
    for q in (ROOT / "quizbank").glob("*.yaml"):
        questions += len(re.findall(r"^  - id: q-", q.read_text(encoding="utf-8"), re.M))
    return {
        "modules": len(modules),
        "concepts": len(concepts),
        "labs": len(labs),
        "exercises": len(exercises),
        "figures": len(figures),
        "questions": questions,
    }


PATTERNS = [
    (r"(\d+)\s*个模块", "modules", "模块数"),
    (r"(\d+)\s*篇概念", "concepts", "概念文档数"),
    (r"(\d+)\s*组习题", "exercises", "习题组数"),
    (r"(\d+)\s*个(?:可运行\s*)?lab", "labs", "lab 数"),
    (r"(\d+)\s*道", "questions", "可判分题数"),
    (r"(\d+)\s*张图", "figures", "生成图数"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="README 规模统计一致性检查")
    ap.add_argument("--quiet", action="store_true", help="只输出问题")
    args = ap.parse_args()

    if not README.exists():
        print("ERROR 找不到 README.md", file=sys.stderr)
        return 1
    text = README.read_text(encoding="utf-8")
    real = actual()

    errors: list[str] = []
    checked = 0
    for pattern, key, label in PATTERNS:
        for m in re.finditer(pattern, text):
            checked += 1
            got = int(m.group(1))
            if got != real[key]:
                line = text[: m.start()].count("\n") + 1
                errors.append(
                    f"README 第 {line} 行：{label}写 {got}，实际 {real[key]}"
                    f"（片段：{text[max(0, m.start()-24):m.end()+8]!r}）"
                )

    for e in errors:
        print(f"ERROR {e}", file=sys.stderr)
    if errors:
        print(f"\n✗ 规模统计检查失败：{len(errors)} 处", file=sys.stderr)
        return 1
    if not args.quiet:
        print("实际数量：" + "、".join(f"{k}={v}" for k, v in real.items()))
    print(f"✓ README 规模统计：{checked} 处数字与仓库一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
