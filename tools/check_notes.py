#!/usr/bin/env python3
"""Validate note structure, wiki links, relative links, and math delimiters."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes"
LECTURE_FILES = sorted(NOTES.glob("Lecture-*.md"))
REQUIRED_HEADINGS = {
    "学习目标",
    "仅作了解",
    "自测",
    "资料定位",
}


def without_code_spans(text: str) -> str:
    return re.sub(r"`[^`]*`", "", text)


def headings(path: Path) -> set[str]:
    return {
        re.sub(r"^#+\s*", "", line).strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("#")
    }


def check_math(path: Path, text: str, errors: list[str]) -> None:
    if r"\(" in text or r"\)" in text or r"\[" in text or r"\]" in text:
        errors.append(f"{path.name}: contains legacy math delimiters")
    if r"\operatorname" in text:
        errors.append(f"{path.name}: uses GitHub-unsupported \\operatorname")

    display_count = sum(line.count("$$") for line in text.splitlines())
    if display_count % 2:
        errors.append(f"{path.name}: unbalanced $$ delimiters")

    for line_no, line in enumerate(text.splitlines(), 1):
        single = line.replace("$$", "")
        if single.count("$") % 2:
            errors.append(f"{path.name}:{line_no}: unbalanced inline $ delimiter")


def main() -> int:
    errors: list[str] = []
    if len(LECTURE_FILES) != 14:
        errors.append(f"expected 14 lecture notes, found {len(LECTURE_FILES)}")

    md_files = sorted(ROOT.rglob("*.md"))
    for path in md_files:
        text = path.read_text(encoding="utf-8")
        body = without_code_spans(text)

        for name, anchor in re.findall(r"\[\[([^]#|]+)(?:#([^]|]+))?", body):
            target = NOTES / f"{name}.md"
            if not target.exists():
                errors.append(f"{path}: missing wiki target {target.name}")
            elif anchor and anchor not in headings(target):
                errors.append(f"{path}: missing wiki anchor {name}#{anchor}")

        for target in re.findall(r"\[[^]]*\]\(([^)]+)\)", body):
            if "://" in target or target.startswith("#"):
                continue
            target_path = (path.parent / target.split("#", 1)[0]).resolve()
            if not target_path.exists():
                errors.append(f"{path}: missing relative link target {target}")

        check_math(path, body, errors)

    for path in LECTURE_FILES:
        text = path.read_text(encoding="utf-8")
        found = headings(path)
        missing = REQUIRED_HEADINGS - found
        if missing:
            errors.append(f"{path.name}: missing headings {sorted(missing)}")
        if "**课程定位**" not in text:
            errors.append(f"{path.name}: missing course-scope tag")
        for level in ("核心掌握", "熟悉", "了解"):
            if level not in text:
                errors.append(f"{path.name}: missing learning level {level}")

    manifest = ROOT / "source_manifest.tsv"
    with manifest.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    if len(rows) != 35:
        errors.append(f"expected 35 manifest rows, found {len(rows)}")
    for row in rows:
        if not (ROOT / row["target_path"]).is_file():
            errors.append(f"missing material: {row['target_path']}")

    checksum_file = ROOT / "checksums.sha256"
    checksum_lines = [
        line
        for line in checksum_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(checksum_lines) != 35:
        errors.append(f"expected 35 checksum rows, found {len(checksum_lines)}")

    if errors:
        print("FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("OK: 14 notes, 35 materials, wiki links, relative links, and math delimiters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

