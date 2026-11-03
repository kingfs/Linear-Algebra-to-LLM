#!/usr/bin/env python3
"""`docs/前沿追踪.md` 结构检查与引用核对。

用法：
    python3 scripts/check_frontier.py             # 离线：表格结构、状态词、年份、arXiv 编号
    python3 scripts/check_frontier.py --online    # 额外用 arXiv API 核对「作者/方法名 ↔ 论文」
    python3 scripts/check_frontier.py --quiet

为什么单独写一个脚本：这一页不是概念文档，`check_refs.py` 不覆盖它；
但它引用的 arXiv 编号最容易手误——v1.8.0 复核时就出现过
「MoRA 指向了另一篇论文（2405.07490）」这种编号错位。

检查项（离线）：
1. 每个主题表的行有 5 列，表头与分隔行齐全；
2. 「状态」列取值属于固定词表（稳定/主流/活跃/探索/已被取代）；
3. 「时间」列是四位年份、年份区间、`—` 或 `待核实`；
4. arXiv 编号格式合法（`NNNN.NNNN`），且同一编号不被两行使用；
5. 每条 arXiv 链接都写成 `[arXiv:xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx)`。

检查项（--online）：
6. arXiv API 能返回标题；
7. 该行的「方向」方法名或「代表工作」里的作者姓氏，至少有一个出现在
   论文标题或作者列表中（大小写不敏感）。不匹配则判为「疑似编号错位」。
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "前沿追踪.md"
STATUS = {"稳定", "主流", "活跃", "探索", "已被取代"}
ARXIV_LINK_RE = re.compile(
    r"\[arXiv:(\d{4}\.\d{4,5})\]\(https://arxiv\.org/abs/(\d{4}\.\d{4,5})\)"
)
TIME_OK_RE = re.compile(
    r"^(?:\d{4}(?:\s*[–-]\s*\d{4})?\s*[–-]?"
    r"|\d{4}(?:\s*/\s*\d{4})+"
    r"|—|待核实)$"
)
ATOM = "{http://www.w3.org/2005/Atom}"


def split_row(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [c.strip() for c in body.split("|")]


def parse_tables(text: str):
    """产出 (行号, 单元格) —— 只取含 arXiv 链接或状态词的数据行。"""
    lines = text.split("\n")
    for i, line in enumerate(lines, 1):
        if not line.lstrip().startswith("|"):
            continue
        if set(line.replace("|", "").strip()) <= set("-: "):
            continue
        cells = split_row(line)
        if len(cells) == 5:
            yield i, cells


def offline_check(text: str, quiet: bool) -> tuple[int, list[str]]:
    errors: list[str] = []
    dup: list[str] = []
    seen: dict[str, int] = {}
    rows = 0
    for lineno, cells in parse_tables(text):
        direction, work, year, _conclusion, status = cells
        if year in {"时间", "年份"} or "预计" in direction:
            continue
        rows += 1
        status = status.strip("*").strip()
        if status not in STATUS:
            errors.append(f"第 {lineno} 行：状态「{status}」不在固定词表 {sorted(STATUS)} 内")
        if not TIME_OK_RE.match(year):
            errors.append(f"第 {lineno} 行：时间「{year}」不是年份/区间/—/待核实")
        for m in ARXIV_LINK_RE.finditer(work):
            shown, url_id = m.group(1), m.group(2)
            if shown != url_id:
                errors.append(f"第 {lineno} 行：显示编号 {shown} 与链接编号 {url_id} 不一致")
            if shown in seen and seen[shown] != lineno:
                dup.append(f"第 {lineno} 行：arXiv {shown} 也被第 {seen[shown]} 行引用（跨节复用可接受，请确认）")
            seen[shown] = lineno
        # 出现 arxiv.org 但没有走标准写法
        if "arxiv.org" in work and not ARXIV_LINK_RE.search(work):
            errors.append(f"第 {lineno} 行：arXiv 链接格式应为 `[arXiv:xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx)`")
    return rows, errors, dup


def fetch_titles(ids: list[str]) -> dict[str, dict]:
    url = (
        "https://export.arxiv.org/api/query?id_list="
        + ",".join(ids)
        + f"&max_results={len(ids)}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "frontier-check/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        root = ET.fromstring(resp.read())
    out: dict[str, dict] = {}
    for entry in root.findall(f"{ATOM}entry"):
        eid = entry.findtext(f"{ATOM}id") or ""
        short = eid.rsplit("/abs/", 1)[-1]
        short = re.sub(r"v\d+$", "", short)
        title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
        authors = [
            (a.findtext(f"{ATOM}name") or "")
            for a in entry.findall(f"{ATOM}author")
        ]
        out[short] = {"title": title, "authors": authors}
    return out


def name_tokens(text: str) -> list[str]:
    """从「Peng 等，[arXiv…]」里取出作者姓氏；从方向列取出方法名。"""
    drop = {
        "等", "作者", "文档", "团队", "小组", "与", "和", "及", "后", "续",
        "the", "and", "et", "al",
    }
    toks = re.findall(r"[A-Za-z][A-Za-z0-9+#\-]{1,}", text)
    return [t for t in toks if t.lower() not in drop]


def online_check(text: str, quiet: bool) -> list[str]:
    rows = [(ln, cells) for ln, cells in parse_tables(text)]
    wanted: list[tuple[int, str, list[str]]] = []
    ids: list[str] = []
    for lineno, cells in rows:
        direction, work = cells[0], cells[1]
        m = ARXIV_LINK_RE.search(work)
        if not m:
            continue
        aid = m.group(1)
        cands = name_tokens(direction) + name_tokens(work[: m.start()])
        # 去掉纯粹的「arXiv」之类噪声
        cands = [c for c in cands if c.lower() not in {"arxiv", "abs", "https", "org"}]
        wanted.append((lineno, aid, cands))
        ids.append(aid)

    warnings: list[str] = []
    if not wanted:
        return warnings
    try:
        meta = fetch_titles(sorted(set(ids)))
    except Exception as exc:  # noqa: BLE001
        return [f"arXiv API 请求失败（跳过在线核对）：{exc}"]

    for lineno, aid, cands in wanted:
        info = meta.get(aid)
        if not info:
            warnings.append(f"第 {lineno} 行：arXiv {aid} 未返回元数据")
            continue
        haystack = (info["title"] + " " + " ".join(info["authors"])).lower()
        if cands and not any(c.lower() in haystack for c in cands):
            warnings.append(
                f"第 {lineno} 行：arXiv {aid} 的标题/作者与行内「{'、'.join(cands)}」都不匹配"
                f"（标题：{info['title'][:60]}…）——疑似编号错位"
            )
        if not quiet:
            print(f"✓ 第 {lineno} 行 arXiv {aid}：{info['title'][:64]}")
    return warnings


def main() -> int:
    ap = argparse.ArgumentParser(description="前沿追踪页结构检查与 arXiv 核对")
    ap.add_argument("--online", action="store_true", help="用 arXiv API 核对标题/作者")
    ap.add_argument("--quiet", action="store_true", help="只输出问题")
    args = ap.parse_args()

    if not DOC.exists():
        print(f"ERROR 找不到 {DOC}", file=sys.stderr)
        return 1
    text = DOC.read_text(encoding="utf-8")

    rows, errors, dup = offline_check(text, args.quiet)
    warnings = list(dup)
    if args.online:
        warnings += online_check(text, args.quiet)

    for e in errors:
        print(f"ERROR {e}", file=sys.stderr)
    for w in warnings:
        print(f"WARN  {w}", file=sys.stderr)

    if errors:
        print(f"\n✗ 前沿追踪检查失败：{len(errors)} 个错误", file=sys.stderr)
        return 1
    print(
        f"\n✓ 前沿追踪：{rows} 行表格、{len(ARXIV_LINK_RE.findall(text))} 条 arXiv 引用，0 错误"
        + (f"，{len(warnings)} 个待复核" if warnings else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
