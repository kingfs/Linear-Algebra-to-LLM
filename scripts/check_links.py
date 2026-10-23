#!/usr/bin/env python3
"""全仓 Markdown 链接检查。

与 check.py 的区别：check.py 只检查概念文档的站内链接；本脚本扫描
README、习题、答案、速查表等所有 .md，捕获相对路径断链。

用法：
    python3 scripts/check_links.py            # 检查全仓
    python3 scripts/check_links.py 11-大模型专题  # 只检查指定目录/文件
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
SKIP_PREFIX = ("http://", "https://", "mailto:", "#", "tel:")
SKIP_DIRS = {".git", "node_modules", "dist", ".astro"}


def ignored(path: Path) -> bool:
    try:
        parts = path.relative_to(ROOT).parts
    except ValueError:
        return False
    if any(part in SKIP_DIRS for part in parts):
        return True
    return "src" in parts and "content" in parts and "generated" in parts


def iter_markdown(paths: list[str]) -> list[Path]:
    if not paths:
        return sorted(p for p in ROOT.rglob("*.md") if not ignored(p))
    files: list[Path] = []
    for raw in paths:
        p = (ROOT / raw).resolve()
        if p.is_dir():
            files.extend(sorted(p.rglob("*.md")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"找不到路径: {raw}", file=sys.stderr)
    return sorted(p for p in set(files) if not ignored(p))


def main() -> int:
    files = iter_markdown(sys.argv[1:])
    errors = 0
    skip_files = {ROOT / "docs" / "模板-概念文档.md"}
    for path in files:
        if path in skip_files:
            continue
        text = path.read_text(encoding="utf-8")
        # 去掉 fenced code，避免把示例里的链接当断链
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        for target in LINK_RE.findall(text):
            if target.startswith(SKIP_PREFIX):
                continue
            rel = target.split("#", 1)[0]
            if not rel:
                continue
            resolved = (path.parent / rel).resolve()
            if not resolved.exists():
                print(f"ERROR {path.relative_to(ROOT)}: 断链 -> {target}")
                errors += 1

    print(f"\n{'✓' if errors == 0 else '✗'} 检查 {len(files)} 个 Markdown 文件，{errors} 个断链")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
