#!/usr/bin/env python3
"""参考文献与致谢（第 11 节）格式与锚点检查。

用法：
    python3 scripts/check_refs.py              # 离线检查全部概念文档
    python3 scripts/check_refs.py 01-向量与几何  # 只检查指定模块
    python3 scripts/check_refs.py --urls       # 额外检查外链可达性
    python3 scripts/check_refs.py --metadata   # 用 Crossref/arXiv 核对 DOI/arXiv 书目
    python3 scripts/check_refs.py --quiet

检查项：
1. 每篇概念文档有第 11 节，且条目数为 2–5；
2. 每条以 `<a id="ref-…">` 开头，含人物/年份、作品名与外链；
3. 页内锚点 id 唯一，正文 `#ref-…` 链接全部可解析；
4. 每条外链指向 http(s)，且不是搜索页/短链；
5. `--urls`：并发请求外链，允许 403/405/429（反爬），404/超时记为失败；
6. `--metadata`：DOI 查 Crossref、arXiv 查官方 API，核对标题是否对得上。

规范见 docs/参考文献写作规范.md。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import json
import re
import sys
import time
from functools import lru_cache
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR_RE = re.compile(r"^(\d{2})-.+$")
CONCEPT_FILE_RE = re.compile(r"^(\d{2})-.+\.md$")
SECTION_RE = re.compile(r"^##\s+11\.\s*参考文献与致谢\s*$", re.M)
ENTRY_RE = re.compile(
    r'^- <a id="(?P<id>ref-[a-z0-9]+(?:-[a-z0-9]+)*)"></a>'
    r"\*\*(?P<who>[^*\n]+)\*\*[：:]\s*(?P<body>.+)$"
)
URL_RE = re.compile(r"\[[^\]]+\]\((https?://(?:[^()\s]|\([^()\s]*\))+)\)")
TITLE_RE = re.compile(r"\*([^*]+)\*")
DOI_RE = re.compile(r"https?://(?:dx\.)?doi\.org/(10\.[^\s)]+)", re.I)
ARXIV_RE = re.compile(r"https?://arxiv\.org/(?:abs|pdf)/([\w.]+)", re.I)
BODY_ANCHOR_RE = re.compile(r"\]\(#(ref-[a-z0-9]+(?:-[a-z0-9]+)*)\)")
YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
BAD_URL_RE = re.compile(r"(google\.[^/]+/search|bing\.com/search|百度|zhihu\.com/search|bit\.ly|t\.cn/)", re.I)
MIN_ENTRIES, MAX_ENTRIES = 2, 5
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
OK_STATUS = {200, 301, 302, 303, 307, 308, 403, 405, 429}


def collect_files(paths: list[str]) -> list[Path]:
    if not paths:
        files: list[Path] = []
        for module_dir in sorted(ROOT.iterdir()):
            if not module_dir.is_dir() or not MODULE_DIR_RE.match(module_dir.name):
                continue
            for path in sorted(module_dir.glob("*.md")):
                if path.name != "README.md" and CONCEPT_FILE_RE.match(path.name):
                    files.append(path)
        return files

    files = []
    for raw in paths:
        p = (ROOT / raw).resolve()
        if p.is_dir():
            files.extend(
                sorted(
                    x
                    for x in p.glob("*.md")
                    if x.name != "README.md" and CONCEPT_FILE_RE.match(x.name)
                )
            )
        elif p.is_file():
            files.append(p)
        else:
            print(f"找不到路径: {raw}", file=sys.stderr)
            raise SystemExit(2)
    return files


def parse_section(text: str) -> tuple[str | None, int, int]:
    """返回 (第 11 节正文, 起始位置, 结束位置)；没有该节时正文为 None。"""
    m = SECTION_RE.search(text)
    if not m:
        return None, -1, -1
    start = m.end()
    nxt = re.search(r"^##\s+", text[start:], re.M)
    end = start + nxt.start() if nxt else len(text)
    return text[start:end], m.start(), end


def check_doc(path: Path, errors: list[str], warnings: list[str]) -> list[dict]:
    rel = path.relative_to(ROOT)
    text = path.read_text(encoding="utf-8")
    section, sec_start, _ = parse_section(text)
    if section is None:
        errors.append(f"{rel}: 缺少「## 11. 参考文献与致谢」")
        return []

    entry_lines = [ln for ln in section.splitlines() if ln.startswith("- ")]
    if not (MIN_ENTRIES <= len(entry_lines) <= MAX_ENTRIES):
        errors.append(
            f"{rel}: 第 11 节应有 {MIN_ENTRIES}–{MAX_ENTRIES} 条文献，实际 {len(entry_lines)} 条"
        )

    ids: list[str] = []
    urls: list[str] = []
    entries: list[dict] = []
    for lineno, line in enumerate(entry_lines, 1):
        m = ENTRY_RE.match(line)
        if not m:
            errors.append(
                f"{rel}: 第 {lineno} 条格式不符（应为 `- <a id=\"ref-…\"></a>**人物（原名, 年份）**：*作品*。说明。[链接](https://…)`）"
            )
            continue
        ref_id = m.group("id")
        who = m.group("who").strip()
        body = m.group("body")
        ids.append(ref_id)
        if not YEAR_RE.search(who):
            warnings.append(f"{rel}: {ref_id} 的署名里没有年份")
        found = URL_RE.findall(body)
        if not found:
            errors.append(f"{rel}: {ref_id} 缺少 http(s) 外链")
        for url in found:
            if BAD_URL_RE.search(url):
                errors.append(f"{rel}: {ref_id} 使用了搜索页/短链 -> {url}")
            urls.append(url)
        title_match = TITLE_RE.search(body)
        entries.append({
            "rel": str(rel),
            "id": ref_id,
            "who": who,
            "title": title_match.group(1).strip() if title_match else "",
            "urls": found,
        })

    dupes = {x for x in ids if ids.count(x) > 1}
    for ref_id in sorted(dupes):
        errors.append(f"{rel}: 锚点 id 重复 -> {ref_id}")

    body_text = text[:sec_start]
    linked = BODY_ANCHOR_RE.findall(body_text)
    for ref_id in sorted(set(linked) - set(ids)):
        errors.append(f"{rel}: 正文链接 #{ref_id} 没有对应文献条目")
    for ref_id in sorted(set(ids) - set(linked)):
        warnings.append(f"{rel}: 文献 {ref_id} 未在正文中链接（可接受，作为延伸阅读）")

    return entries


def check_url(url: str) -> tuple[str, str]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            status = resp.status
        if status in OK_STATUS:
            return url, ""
        if status in {404, 410}:
            return url, f"HTTP {status}"
        return url, ""
    except urllib.error.HTTPError as exc:
        if exc.code in OK_STATUS:
            return url, ""
        return url, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return url, f"网络错误: {exc.reason}"
    except TimeoutError:
        return url, "超时"
    except Exception as exc:  # noqa: BLE001 - 检查脚本要报告而不是崩溃
        return url, f"{type(exc).__name__}: {exc}"


def norm_title(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
    return " ".join(text.split())


def title_matches(cited: str, actual: str) -> bool:
    if not cited or not actual:
        return True
    if re.search(r"[\u4e00-\u9fff]", cited):
        return True  # 中文译名无法与原文标题直接比对
    a, b = norm_title(cited), norm_title(actual)
    if a in b or b in a:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.55


def fetch_with_retry(url: str, *, accept: str | None = None) -> str:
    """抓取文本，对 429/5xx 与网络错误做有限重试。"""
    headers = {"User-Agent": UA}
    if accept:
        headers["Accept"] = accept
    last: Exception | None = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                raise
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
        time.sleep(1.5 * (attempt + 1))
    assert last is not None
    raise last


@lru_cache(maxsize=512)
def fetch_json(url: str):
    return json.loads(fetch_with_retry(url, accept="application/json"))


@lru_cache(maxsize=512)
def fetch_text(url: str) -> str:
    return fetch_with_retry(url)


def check_metadata(entry: dict) -> list[str]:
    """核对 DOI / arXiv 的标题；返回错误列表（元数据对不上才算错误）。"""
    problems: list[str] = []
    for url in entry["urls"]:
        doi_m = DOI_RE.search(url)
        arxiv_m = ARXIV_RE.search(url)
        try:
            if doi_m:
                doi = urllib.parse.unquote(doi_m.group(1))
                data = fetch_json("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe=""))
                msg = data["message"]
                actual = (msg.get("title") or [""])[0]
                if not title_matches(entry["title"], actual):
                    problems.append(
                        f"{entry['rel']}: {entry['id']} 的 DOI 标题对不上："
                        f"引用「{entry['title']}」，Crossref 为「{actual}」"
                    )
            elif arxiv_m:
                arxiv_id = arxiv_m.group(1)
                xml = fetch_text("https://export.arxiv.org/api/query?id_list=" + arxiv_id)
                root = ET.fromstring(xml)
                ns = {"a": "http://www.w3.org/2005/Atom"}
                node = root.find("a:entry/a:title", ns)
                actual = (node.text or "").strip() if node is not None else ""
                if actual and not title_matches(entry["title"], actual):
                    problems.append(
                        f"{entry['rel']}: {entry['id']} 的 arXiv 标题对不上："
                        f"引用「{entry['title']}」，arXiv 为「{actual}」"
                    )
        except Exception as exc:  # noqa: BLE001 - 网络问题只警告
            problems.append(f"__warn__{entry['rel']}: {entry['id']} 无法核对元数据（{exc}）")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="参考文献格式与锚点检查")
    parser.add_argument("paths", nargs="*", help="只检查指定模块目录/文件")
    parser.add_argument("--urls", action="store_true", help="检查外链可达性")
    parser.add_argument("--metadata", action="store_true", help="用 Crossref/arXiv 核对 DOI/arXiv 标题")
    parser.add_argument("--quiet", action="store_true", help="只输出错误")
    args = parser.parse_args()

    files = collect_files(args.paths)
    errors: list[str] = []
    warnings: list[str] = []
    all_urls: set[str] = set()
    entries: list[dict] = []
    for path in files:
        entries.extend(check_doc(path, errors, warnings))
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("- "):
                all_urls.update(URL_RE.findall(line))
    total_entries = len(entries)

    if args.metadata and entries:
        checked = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for problems in pool.map(check_metadata, entries):
                for problem in problems:
                    if problem.startswith("__warn__"):
                        warnings.append(problem.replace("__warn__", "元数据: ", 1))
                    else:
                        errors.append(problem)
                checked += 1

    if args.urls and all_urls:
        failed: list[tuple[str, str]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for url, reason in pool.map(check_url, sorted(all_urls)):
                if reason:
                    failed.append((url, reason))
        for url, reason in sorted(failed):
            line = f"外链不可用（{reason}）-> {url}"
            if reason.startswith(("HTTP 404", "HTTP 410")):
                errors.append(line)
            else:
                warnings.append("网络波动，稍后复测: " + line)

    for w in warnings:
        if not args.quiet:
            print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")

    if errors:
        print(f"\n✗ {len(files)} 篇文档，{len(errors)} 个错误，{len(warnings)} 个警告")
        return 1
    if not args.quiet:
        extra = f"，{checked} 条 DOI/arXiv 元数据已核对" if args.metadata else ""
        print(
            f"\n✓ {len(files)} 篇文档，{total_entries} 条文献，"
            f"{len(all_urls)} 个外链{extra}，0 错误（{len(warnings)} 个警告）"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
