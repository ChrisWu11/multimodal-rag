#!/usr/bin/env python3
"""Run static completeness and consistency checks on the final-report LaTeX source."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9'-]*\b")
ENTRY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)", re.IGNORECASE)
CITE_RE = re.compile(r"\\(?:textcite|parencite|cite)\s*\{([^}]+)\}")
LABEL_RE = re.compile(r"\\label\s*\{([^}]+)\}")
REF_RE = re.compile(r"\\(?:ref|pageref|autoref)\s*\{([^}]+)\}")
GRAPHIC_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\s*\{([^}]+)\}")


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def remove_environment(text: str, environment: str) -> str:
    return re.sub(
        rf"\\begin\{{{environment}\}}.*?\\end\{{{environment}\}}",
        " ",
        text,
        flags=re.DOTALL,
    )


def visible_text(text: str, *, remove_floats: bool) -> str:
    text = strip_comments(text)
    text = remove_environment(text, "tikzpicture")
    if remove_floats:
        for environment in ("figure", "table"):
            text = remove_environment(text, environment)
    text = re.sub(r"\$.*?\$", " ", text, flags=re.DOTALL)
    text = re.sub(
        r"\\(?:cite|textcite|parencite|label|ref|pageref|autoref|path|url)"
        r"(?:\[[^\]]*\])?\{[^}]*\}",
        " ",
        text,
    )
    text = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}", " ", text)
    text = re.sub(r"\\begin\{[^}]+\}|\\end\{[^}]+\}", " ", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace(r"\&", "&").replace(r"\%", "%").replace(r"\_", "_")
    text = text.replace("{", " ").replace("}", " ")
    return re.sub(r"\s+", " ", text)


def parse_bib_keys(text: str) -> set[str]:
    return {match.strip() for match in ENTRY_RE.findall(text)}


def parse_cite_keys(text: str) -> set[str]:
    return {
        key.strip()
        for group in CITE_RE.findall(text)
        for key in group.split(",")
        if key.strip()
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--bib", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tex_path = args.tex.expanduser().resolve()
    bib_path = args.bib.expanduser().resolve()
    tex = tex_path.read_text(encoding="utf-8")
    bib = bib_path.read_text(encoding="utf-8")
    report_start = tex.index(r"\section*{Abstract}")
    report_end = tex.index(r"\section*{Acknowledgements}")
    report = tex[report_start:report_end]
    prose_words = len(WORD_RE.findall(visible_text(report, remove_floats=True)))
    inclusive_words = len(WORD_RE.findall(visible_text(report, remove_floats=False)))

    bib_keys = parse_bib_keys(bib)
    cite_keys = parse_cite_keys(tex)
    labels = LABEL_RE.findall(tex)
    refs = REF_RE.findall(tex)
    duplicate_labels = sorted(
        label for label in set(labels) if labels.count(label) > 1
    )
    missing_refs = sorted(set(refs) - set(labels))
    missing_bib_entries = sorted(cite_keys - bib_keys)
    uncited_bib_entries = sorted(bib_keys - cite_keys)

    graphics = GRAPHIC_RE.findall(tex)
    graphic_dirs = [
        tex_path.parent / value
        for group in re.findall(r"\\graphicspath\{((?:\{[^}]+\})+)\}", tex)
        for value in re.findall(r"\{([^}]+)\}", group)
    ]
    if not graphic_dirs:
        graphic_dirs = [tex_path.parent]
    missing_graphics = []
    for graphic in graphics:
        candidates = [directory / graphic for directory in graphic_dirs]
        if any(candidate.exists() for candidate in candidates):
            continue
        if not Path(graphic).suffix and any(
            list(candidate.parent.glob(f"{candidate.name}.*"))
            for candidate in candidates
        ):
            continue
        if not any(candidate.exists() for candidate in candidates):
            missing_graphics.append(graphic)

    keyword_match = re.search(
        r"\\textbf\{Keywords:\}\s*([^\n]+)",
        tex,
    )
    keywords = (
        [value.strip() for value in keyword_match.group(1).split(";")]
        if keyword_match
        else []
    )
    placeholders = sorted(
        {
            token
            for token in ("AUTHOR GUIDANCE", "[[", "TODO", "FIXME", "Replace this")
            if token in tex
        }
    )
    required_cover = [
        "Surname & \\textbf{Wu}",
        "First Name & \\textbf{Kang}",
        "3035412",
        "Jiaqi Ye",
        (
            "Development and Evaluation of an Evidence-Grounded Multimodal "
            "RAG System for Scientific Literature"
        ),
    ]
    missing_cover = [value for value in required_cover if value not in tex]
    required_sections = [
        "Introduction",
        "Literature Review and State of the Art",
        "Methodology and System Design",
        "Results",
        "Discussion",
        "Impact Statement",
        "Conclusions",
    ]
    missing_sections = [
        section
        for section in required_sections
        if not re.search(
            rf"\\section\{{{re.escape(section)}\}}",
            tex,
        )
    ]
    figure_count = len(re.findall(r"\\begin\{figure\}", tex))
    table_count = len(re.findall(r"\\begin\{table\}", tex))
    checks = {
        "prose_words_le_5000": prose_words <= 5000,
        "inclusive_words_le_5000": inclusive_words <= 5000,
        "exactly_five_keywords": len(keywords) == 5,
        "figures_le_10": figure_count <= 10,
        "tables_le_10": table_count <= 10,
        "twenty_five_references": len(bib_keys) == 25,
        "all_citations_resolve": not missing_bib_entries,
        "all_bibliography_entries_cited": not uncited_bib_entries,
        "all_cross_references_resolve": not missing_refs,
        "no_duplicate_labels": not duplicate_labels,
        "all_graphics_exist": not missing_graphics,
        "cover_complete": not missing_cover,
        "required_sections_present": not missing_sections,
        "no_placeholders": not placeholders,
    }
    payload = {
        "tex": str(tex_path),
        "bib": str(bib_path),
        "counts": {
            "prose_words_excluding_floats": prose_words,
            "inclusive_words_with_floats": inclusive_words,
            "keywords": len(keywords),
            "figures": figure_count,
            "tables": table_count,
            "bibliography_entries": len(bib_keys),
            "cited_entries": len(cite_keys),
        },
        "checks": checks,
        "details": {
            "keywords": keywords,
            "missing_bib_entries": missing_bib_entries,
            "uncited_bib_entries": uncited_bib_entries,
            "missing_cross_references": missing_refs,
            "duplicate_labels": duplicate_labels,
            "missing_graphics": missing_graphics,
            "missing_cover_values": missing_cover,
            "missing_sections": missing_sections,
            "placeholders": placeholders,
        },
    }
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
