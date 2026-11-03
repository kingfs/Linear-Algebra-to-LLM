#!/usr/bin/env python3
"""模块 README 阅读时间一致性检查。

用法：
    python3 scripts/check_minutes.py
    python3 scripts/check_minutes.py --quiet

背景：`NN-概念名.md` 的 front matter `minutes` 是「阅读 + 练习」分钟数，
模块 `README.md` 的「分次阅读建议」必须与它一致，否则会出现
「README 说 300 分钟、逐篇相加却是 340 分钟」这类自相矛盾。

约定：每个含「分次阅读建议」的模块 README，在该小节内放一行机器可读注释：

    <!-- reading-minutes: docs=255 extra=0 total=255 -->

- `docs` ：该模块所有 `NN-*.md` 的 `minutes` 之和；
- `extra`：不属于概念文档的附加时间（例如 M13 手册 §3–§11 的 60 分钟）；
- `total`：正文展示给读者的数字，必须等于 `docs + extra`。

检查项：
1. 注释存在且三个字段齐全、可解析；
2. `docs` 等于 front matter 实际求和；
3. `total == docs + extra`；
4. 正文「预计阅读时间……：N 分钟」中的 N 等于 `total`。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR_RE = re.compile(r"^\d{2}-.+$")
CONCEPT_FILE_RE = re.compile(r"^\d{2}-.+\.md$")
MINUTES_FM_RE = re.compile(r"^minutes:\s*(\d+)\s*$", re.M)
ANNOTATION_RE = re.compile(
    r"<!--\s*reading-minutes:\s*docs=(\d+)\s+extra=(\d+)\s+total=(\d+)\s*-->"
)
SECTION_RE = re.compile(r"^##\s+(?:\d+(?:\.\d+)?\s+)?分次阅读建议\s*$", re.M)
PROSE_TOTAL_RE = re.compile(r"预计阅读时间[^\n]*?[：:]\s*(\d+)\s*分钟")


def front_matter_minutes(path: Path) -> int | None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    fm = text[3:end]
    m = MINUTES_FM_RE.search(fm)
    return int(m.group(1)) if m else None


def section_text(text: str) -> str | None:
    m = SECTION_RE.search(text)
    if not m:
        return None
    start = m.end()
    nxt = re.search(r"^##\s", text[start:], re.M)
    return text[start : start + nxt.start()] if nxt else text[start:]


def check(quiet: bool) -> int:
    errors: list[str] = []
    checked = 0

    for d in sorted(ROOT.iterdir()):
        if not d.is_dir() or not MODULE_DIR_RE.match(d.name):
            continue
        readme = d / "README.md"
        if not readme.exists():
            continue
        text = readme.read_text(encoding="utf-8")
        sec = section_text(text)
        if sec is None:
            continue
        checked += 1
        rel = readme.relative_to(ROOT)

        docs = 0
        for f in sorted(d.glob("*.md")):
            if not CONCEPT_FILE_RE.match(f.name):
                continue
            v = front_matter_minutes(f)
            if v is not None:
                docs += v

        m = ANNOTATION_RE.search(sec)
        if not m:
            errors.append(
                f"{rel}: 「分次阅读建议」缺少 `<!-- reading-minutes: docs=… extra=… total=… -->` 注释"
            )
            continue
        a_docs, a_extra, a_total = (int(x) for x in m.groups())

        if a_docs != docs:
            errors.append(
                f"{rel}: docs 标注 {a_docs}，但 front matter `minutes` 实际求和为 {docs}"
            )
        if a_total != a_docs + a_extra:
            errors.append(
                f"{rel}: total 标注 {a_total} ≠ docs({a_docs}) + extra({a_extra}) = {a_docs + a_extra}"
            )

        pm = PROSE_TOTAL_RE.search(sec)
        if not pm:
            errors.append(f"{rel}: 找不到「预计阅读时间……：N 分钟」")
        elif int(pm.group(1)) != a_total:
            errors.append(
                f"{rel}: 正文写 {pm.group(1)} 分钟，与 total 标注 {a_total} 不一致"
            )

        if not quiet and not errors:
            print(f"✓ {rel}: docs={docs} extra={a_extra} total={a_total}")

    if errors:
        for e in errors:
            print(f"ERROR {e}", file=sys.stderr)
        print(f"\n✗ 阅读时间检查失败：{len(errors)} 个问题", file=sys.stderr)
        return 1
    print(f"\n✓ {checked} 个模块 README 的阅读时间与 front matter 一致")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="模块 README 阅读时间一致性检查")
    ap.add_argument("--quiet", action="store_true", help="只输出错误")
    args = ap.parse_args()
    return check(args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
