#!/usr/bin/env python3
"""课程仓库质量门禁。

用法：
    python3 scripts/check.py            # 检查全部
    python3 scripts/check.py --quiet    # 只输出错误
    python3 scripts/check.py 01-向量与几何   # 只检查某个目录

检查项：
1. 概念文档 front matter 必填字段与取值；
2. id 全局唯一、module 与目录一致、prereq 指向存在的 id；
3. 固定十个二级小节齐全且顺序正确；
4. 「通往大模型」内容不少于 80 字；
5. 站内相对链接可解析；
6. 无 TODO/TBD/待补充 等占位符；
7. UTF-8 编码。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR_RE = re.compile(r"^(\d{2})-(.+)$")
CONCEPT_FILE_RE = re.compile(r"^(\d{2})-(.+)\.md$")
REQUIRED_FIELDS = {
    "id", "title", "module", "slug", "prereq",
    "minutes", "level", "tags", "llm_hooks",
}
LEVELS = {"核心", "进阶", "选学"}
HEADINGS = [
    "场景引入", "直觉", "定义", "为什么这样定义", "性质与推导",
    "例子", "常见误区", "概念网络", "通往大模型", "练习", "一页速记",
]
# 0. 场景引入 / 1. 直觉 / ... / 9. 练习 / 10. 一页速记
EXPECTED_NUMBERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
PLACEHOLDER_RE = re.compile(r"\b(TODO|TBD|FIXME|XXX)\b|待补充|此处占位|placeholder", re.I)
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def read_text(path: Path, errors: list[str]) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        errors.append(f"{path}: 不是合法的 UTF-8 文件")
    except OSError as exc:
        errors.append(f"{path}: 读取失败 {exc}")
    return None


def parse_front_matter(text: str) -> tuple[dict[str, str], int] | None:
    if not text.startswith("---"):
        return None
    lines = text.splitlines()
    if lines[0].strip() != "---":
        return None
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return None
    data: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        data[key.strip()] = value
    return data, end + 1


def strip_inline_list(value: str) -> list[str]:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if not value.strip():
        return []
    return [item.strip().strip("'\"") for item in value.split(",") if item.strip()]


def collect_concepts() -> list[Path]:
    files: list[Path] = []
    for module_dir in sorted(ROOT.iterdir()):
        if not module_dir.is_dir() or not MODULE_DIR_RE.match(module_dir.name):
            continue
        for path in sorted(module_dir.glob("*.md")):
            if path.name == "README.md":
                continue
            if CONCEPT_FILE_RE.match(path.name):
                files.append(path)
    return files


def check_doc(path: Path, all_ids: dict[str, Path], errors: list[str], warnings: list[str]) -> None:
    text = read_text(path, errors)
    if text is None:
        return
    rel = path.relative_to(ROOT)
    parsed = parse_front_matter(text)
    if parsed is None:
        errors.append(f"{rel}: 缺少 YAML front matter")
        return
    fm, body_start = parsed

    missing = REQUIRED_FIELDS - fm.keys()
    if missing:
        errors.append(f"{rel}: front matter 缺少字段 {sorted(missing)}")
    extra = fm.keys() - REQUIRED_FIELDS
    if extra:
        warnings.append(f"{rel}: front matter 有额外字段 {sorted(extra)}")

    concept_id = fm.get("id", "")
    if not re.fullmatch(r"\d+\.\d+", concept_id):
        errors.append(f"{rel}: id '{concept_id}' 应形如 1.3")

    module = fm.get("module", "")
    dir_match = MODULE_DIR_RE.match(path.parent.name)
    if dir_match:
        expected_module = "M" + dir_match.group(1)
        if module != expected_module:
            errors.append(f"{rel}: module '{module}' 与目录不符（应为 {expected_module}）")

    level = fm.get("level", "")
    if level not in LEVELS:
        errors.append(f"{rel}: level '{level}' 必须是 {sorted(LEVELS)} 之一")

    minutes = fm.get("minutes", "")
    if not minutes.isdigit() or not (5 <= int(minutes) <= 600):
        errors.append(f"{rel}: minutes '{minutes}' 应为 5–600 的整数")

    for field in ("title", "slug"):
        if not fm.get(field):
            errors.append(f"{rel}: {field} 不能为空")
    if fm.get("slug") and not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", fm["slug"]):
        errors.append(f"{rel}: slug '{fm['slug']}' 只能是小写字母/数字/连字符")

    for field in ("tags", "llm_hooks", "prereq"):
        if field in fm and not fm[field].startswith("["):
            warnings.append(f"{rel}: {field} 建议写成行内列表 [...]")

    body = "\n".join(text.splitlines()[body_start:])
    if PLACEHOLDER_RE.search(body):
        errors.append(f"{rel}: 含 TODO/待补充 等占位符")

    # 小节检查
    headings = [(int(m.group(1)), m.group(2), m.start())
                for m in re.finditer(r"^##\s+(\d+)\.\s*(.+?)\s*$", body, re.M)]
    numbers = [h[0] for h in headings]
    if numbers != EXPECTED_NUMBERS:
        errors.append(f"{rel}: 二级小节编号应依次为 {EXPECTED_NUMBERS}，实际为 {numbers}")
    for idx, (num, title, _pos) in enumerate(headings):
        if num < len(HEADINGS) and not title.startswith(HEADINGS[num]):
            errors.append(f"{rel}: 第 {num} 节标题应以「{HEADINGS[num]}」开头，实际为「{title}」")

    # 「通往大模型」非空
    llm_idx = next((i for i, (n, _, _) in enumerate(headings) if n == 8), None)
    if llm_idx is not None:
        start = headings[llm_idx][2]
        end = headings[llm_idx + 1][2] if llm_idx + 1 < len(headings) else len(body)
        section = body[start:end]
        content = re.sub(r"^##.*$", "", section, flags=re.M)
        content = re.sub(r"\s+", "", content)
        if len(content) < 80:
            errors.append(f"{rel}: 「通往大模型」内容过短（{len(content)} 字，至少 80）")

    # 链接检查
    for raw in LINK_RE.findall(body):
        if raw.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = unquote(raw.split("#", 1)[0])
        if not target:
            continue
        resolved = (path.parent / target).resolve()
        try:
            resolved.relative_to(ROOT.resolve())
        except ValueError:
            continue
        if not resolved.exists():
            errors.append(f"{rel}: 断链 -> {raw}")


def main() -> int:
    parser = argparse.ArgumentParser(description="课程仓库质量门禁")
    parser.add_argument("paths", nargs="*", help="只检查指定模块目录")
    parser.add_argument("--quiet", action="store_true", help="只输出错误")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    all_ids: dict[str, Path] = {}

    # 全局 id 注册表（用于跨模块 prereq 与重复检测）
    all_concepts = collect_concepts()
    for path in all_concepts:
        text = read_text(path, errors)
        if text is None:
            continue
        parsed = parse_front_matter(text)
        if not parsed:
            continue
        concept_id = parsed[0].get("id", "")
        if not re.fullmatch(r"\d+\.\d+", concept_id):
            continue
        if concept_id in all_ids:
            errors.append(
                f"{path.relative_to(ROOT)}: id {concept_id} 与 "
                f"{all_ids[concept_id].relative_to(ROOT)} 重复"
            )
        else:
            all_ids[concept_id] = path

    if args.paths:
        files: list[Path] = []
        for raw in args.paths:
            p = (ROOT / raw).resolve()
            if p.is_dir():
                files.extend(sorted(x for x in p.glob("*.md")
                                    if x.name != "README.md" and CONCEPT_FILE_RE.match(x.name)))
            elif p.is_file():
                files.append(p)
            else:
                print(f"找不到路径: {raw}", file=sys.stderr)
                return 2
    else:
        files = all_concepts

    for path in files:
        check_doc(path, all_ids, errors, warnings)

    # prereq 存在性（全部文档解析完后统一检查）
    for path in files:
        text = read_text(path, errors)
        if text is None:
            continue
        parsed = parse_front_matter(text)
        if not parsed:
            continue
        fm = parsed[0]
        for dep in strip_inline_list(fm.get("prereq", "")):
            if dep and dep not in all_ids:
                rel = path.relative_to(ROOT)
                warnings.append(f"{rel}: 前置概念 {dep} 尚未创建")

    for w in warnings:
        if not args.quiet:
            print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")

    total = len(files)
    if errors:
        print(f"\n✗ {total} 篇文档，{len(errors)} 个错误，{len(warnings)} 个警告")
        return 1
    if not args.quiet:
        print(f"\n✓ {total} 篇概念文档全部通过检查（{len(warnings)} 个警告）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
