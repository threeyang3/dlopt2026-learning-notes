#!/usr/bin/env python3
"""Validate note structure, wiki links, relative links, and math delimiters."""

from __future__ import annotations

import csv
import json
import os
import hashlib
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
        delimiters = list(re.finditer(r"(?<!\\)\$(?!\$)", single))
        for opening, closing in zip(delimiters[0::2], delimiters[1::2]):
            if opening.start() and single[opening.start() - 1] in "，。、；：）":
                errors.append(
                    f"{path.name}:{line_no}: inline math after CJK punctuation needs a space"
                )


def markdown_files(root: Path) -> list[Path]:
    """Check maintained Markdown, not vendored dependencies or generated copies."""
    generated = {".git", "node_modules", "dist", "test-results", "playwright-report", "__pycache__"}
    result: list[Path] = []
    for current, directories, files in os.walk(root):
        directories[:] = [name for name in directories if name not in generated]
        result.extend(Path(current) / name for name in files if name.endswith(".md"))
    return sorted(result)


def main() -> int:
    errors: list[str] = []
    if len(LECTURE_FILES) != 14:
        errors.append(f"expected 14 lecture notes, found {len(LECTURE_FILES)}")

    supplements = json.loads((ROOT / "website/src/content/supplements.json").read_text(encoding="utf-8"))
    supplement_pdfs = {entry["pdf"] for entry in supplements}
    supplement_notes = {entry["note"] for entry in supplements}
    if len(supplement_pdfs) != len(supplements) or len(supplement_notes) != len(supplements):
        errors.append("duplicate supplemental PDF/note registration")
    actual_supplements = {str(path.relative_to(ROOT)) for path in NOTES.glob("Supplement-*.md")}
    if actual_supplements != supplement_notes:
        errors.append("supplemental notes differ from explicit content index")
    for entry in supplements:
        if not (ROOT / entry["note"]).is_file():
            errors.append(f"missing supplement note: {entry['note']}")
        target = ROOT / entry["pdf"]
        if target.is_file() and (target.stat().st_size != entry["bytes"] or hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]):
            errors.append(f"supplement differs from source identity: {entry['pdf']}")

    md_files = markdown_files(ROOT)
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

    for path in LECTURE_FILES + sorted(NOTES.glob("Supplement-*.md")):
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
    expected_count = 35 + len(supplements)
    core_rows = [row for row in rows if row["lecture"].startswith("Lecture-")]
    supplemental_rows = [row for row in rows if not row["lecture"].startswith("Lecture-")]
    if len(core_rows) != 35 or len(rows) != expected_count:
        errors.append(f"expected 35 core + {len(supplements)} supplemental manifest rows, found {len(rows)}")
    if {row["target_path"] for row in supplemental_rows} != supplement_pdfs:
        errors.append("supplemental manifest paths differ from explicit content index")
    manifest_paths = {row["target_path"] for row in rows}
    if len(manifest_paths) != len(rows):
        errors.append("duplicate manifest target")
    if manifest_paths != {str(path.relative_to(ROOT)) for path in (ROOT / "materials").rglob("*.pdf")}:
        errors.append("PDF files differ from registered manifest")
    for row in rows:
        if not (ROOT / row["target_path"]).is_file():
            errors.append(f"missing material: {row['target_path']}")

    checksum_file = ROOT / "checksums.sha256"
    checksum_lines = [
        line
        for line in checksum_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(checksum_lines) != expected_count:
        errors.append(f"expected {expected_count} checksum rows, found {len(checksum_lines)}")
    checksum_paths = [line.split(maxsplit=1)[1].strip() for line in checksum_lines if len(line.split(maxsplit=1)) == 2]
    if len(set(checksum_paths)) != len(checksum_paths) or set(checksum_paths) != manifest_paths:
        errors.append("checksum paths must match manifest exactly, without duplicates")
    for entry in checksum_lines:
        parts = entry.split(maxsplit=1)
        if len(parts) != 2:
            errors.append(f"invalid checksum row: {entry}")
            continue
        expected, relative_path = parts
        target = ROOT / relative_path.strip()
        if not target.is_file():
            errors.append(f"checksum target missing: {relative_path}")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f"checksum mismatch: {relative_path}")

    if errors:
        print("FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        f"OK: 14 core notes + {len(supplements)} supplements, {expected_count} materials, links, math delimiters, and SHA-256 checksums"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

