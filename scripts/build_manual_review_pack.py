#!/usr/bin/env python3
"""Prepare reviewer-friendly CSVs for the human checks that cannot be automated."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def clean_tex(text: str) -> str:
    text = re.sub(r"\\(?:code|path|textbf|emph)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:cite|ref|cref)\{[^{}]*\}", "", text)
    text = re.sub(r"\\[A-Za-z*]+(?:\[[^]]*\])?", "", text)
    return " ".join(text.replace("{", "").replace("}", "").split())


def citation_claim_rows(tex_path: Path) -> list[dict[str, Any]]:
    text = tex_path.read_text(encoding="utf-8")
    current_section = ""
    rows = []
    paragraph: list[str] = []
    paragraph_start = 1

    def flush() -> None:
        nonlocal paragraph
        joined = " ".join(paragraph).strip()
        keys = [
            key.strip()
            for group in re.findall(r"\\cite\{([^}]+)\}", joined)
            for key in group.split(",")
            if key.strip()
        ]
        if keys:
            rows.append(
                {
                    "section": current_section,
                    "source_line": paragraph_start,
                    "citation_keys": "; ".join(dict.fromkeys(keys)),
                    "claim_text": clean_tex(joined),
                    "metadata_verified": "yes",
                    "source_read": "",
                    "claim_supported": "",
                    "scope_accurate": "",
                    "review_status": "pending",
                    "reviewer_notes": "",
                }
            )
        paragraph = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        section = re.match(r"\\(?:section|subsection)\*?\{([^}]+)\}", line)
        if section:
            flush()
            current_section = section.group(1)
        if not line.strip():
            flush()
            paragraph_start = line_number + 1
        elif not line.lstrip().startswith("%"):
            if not paragraph:
                paragraph_start = line_number
            paragraph.append(line.strip())
    flush()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", type=Path, required=True)
    parser.add_argument("--pdf-sample", type=Path, required=True)
    parser.add_argument("--image-audit", type=Path, required=True)
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    silver_rows = []
    for row in load_jsonl(args.silver):
        sources = row.get("gold_sources") or []
        silver_rows.append(
            {
                "id": row.get("id"),
                "type": row.get("type"),
                "question": row.get("question"),
                "gold_answer_excerpt": str(row.get("gold_answer") or "")[:500],
                "gold_source_titles": "; ".join(
                    str(source.get("title") or "") for source in sources
                ),
                "gold_source_dois": "; ".join(
                    str(source.get("doi") or "") for source in sources
                ),
                "gold_chunk_ids": "; ".join(
                    str(source.get("chunk_id") or "") for source in sources
                ),
                "question_clear": "",
                "source_contains_answer": "",
                "answer_matches_source": "",
                "scope_safe": "",
                "decision": "pending",
                "corrected_question": "",
                "corrected_gold_source": "",
                "reviewer_notes": "",
            }
        )
    write_csv(args.output / "01-silver-question-review.csv", silver_rows)

    pdf_sample = json.loads(args.pdf_sample.read_text(encoding="utf-8"))
    pdf_rows = [
        {
            "doc_id": row.get("doc_id"),
            "title": row.get("title"),
            "year": row.get("year"),
            "page_count": row.get("page_count"),
            "extracted_pages": row.get("extracted_pages"),
            "mean_chars_per_page": row.get("mean_chars_per_page"),
            "licence": row.get("licence"),
            "pdf_path": row.get("pdf_path"),
            "reading_order_correct": "",
            "pages_complete": "",
            "headings_and_sections_correct": "",
            "header_footer_noise_acceptable": "",
            "formula_table_corruption_acceptable": "",
            "metadata_correct": "",
            "decision": "pending",
            "reviewer_notes": "",
        }
        for row in pdf_sample
    ]
    write_csv(args.output / "02-pdf-technical-review.csv", pdf_rows)

    image_audit = json.loads(args.image_audit.read_text(encoding="utf-8"))
    image_rows = [
        {
            "candidate_id": row.get("candidate_id"),
            "title": row.get("title"),
            "doi": row.get("doi"),
            "page": row.get("page"),
            "figure_number": row.get("figure_number"),
            "article_licence": row.get("article_licence"),
            "licence_url": row.get("licence_url"),
            "third_party_caption_signal": row.get("third_party_caption_signal"),
            "original_pdf_notice_checked": "",
            "no_third_party_panel": "",
            "attribution_approved": "",
            "decision": "pending",
            "reviewer_notes": "",
        }
        for row in image_audit["figures"]
    ]
    write_csv(args.output / "03-image-rights-review.csv", image_rows)

    claim_rows = citation_claim_rows(args.tex)
    write_csv(args.output / "04-claim-citation-support-review.csv", claim_rows)

    readme = """# Human Review Pack

These four CSV files isolate the checks that require a person to read the
source material. Automated experiments and metadata checks must not be used to
pre-fill these judgement columns.

1. Review all 50 silver questions. Use `approved`, `corrected`, or `excluded`
   in `decision`.
2. Inspect the 30 sampled PDFs page by page and record parser defects.
3. Open each of the seven source papers behind the 12 figures and confirm that
   no panel is credited to a third party despite the article-level CC BY 4.0
   licence.
4. Read the cited source for every report claim and confirm that the wording,
   population, conditions and limitations are supported.

Keep the completed files with the final evaluation archive. Update the report
only after recording the decisions; do not silently edit the original silver
manifest.
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    print(
        f"Prepared {len(silver_rows)} question rows, {len(pdf_rows)} PDF rows, "
        f"{len(image_rows)} image rows and {len(claim_rows)} citation-claim rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
