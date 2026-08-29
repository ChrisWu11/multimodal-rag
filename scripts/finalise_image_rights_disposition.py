#!/usr/bin/env python3
"""Record a conservative public-release disposition for image-QA source figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FLAGGED_ACTIONS = {
    "doc_0150_4731780ae890_p011_f01": {
        "public_disposition": "exclude_pending_prior-study-rights-confirmation",
        "reason": (
            "The current article is CC BY, but the lesion photographs originated in a cited "
            "prior study. Retain aggregate evaluation only; do not reproduce the panel."
        ),
    },
    "doc_0010_12273d2ffc1c_p007_f01": {
        "public_disposition": "exclude_biorender-labelled-figure",
        "reason": (
            "The caption states Created with BioRender.com. Article-level CC BY does not by "
            "itself establish rights to redistribute BioRender assets."
        ),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precheck", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    with args.precheck.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    output_rows = []
    for row in rows:
        candidate_id = row["candidate_id"]
        flagged = FLAGGED_ACTIONS.get(candidate_id)
        output_rows.append(
            {
                "candidate_id": candidate_id,
                "doi": row["doi"],
                "figure_number": row["figure_number"],
                "article_licence_verified": row["article_licence_verified"],
                "source_page_precheck": row["ai_precheck_outcome"],
                "used_for_internal_evaluation": "yes",
                "source_image_in_final_report": "no",
                "source_image_in_public_package": "no",
                "public_disposition": (
                    flagged["public_disposition"]
                    if flagged
                    else "not-redistributed-aggregate-results-only"
                ),
                "reason": (
                    flagged["reason"]
                    if flagged
                    else (
                        "The source image is not redistributed. Any future reproduction must "
                        "retain article, author, DOI and CC BY attribution."
                    )
                ),
                "independent_rights_holder_confirmation": "not_performed",
                "reviewer_type": "AI-assisted rights-risk disposition; not legal advice",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    summary_lines = [
        "# Image Rights Public-Release Disposition",
        "",
        "All 12 source figures remain usable only for the project's internal technical",
        "evaluation. No source image is reproduced in the final report or included in the",
        "public reproducibility package; only aggregate metrics are released.",
        "",
        "Two figures receive stricter exclusions:",
        "",
        "- PLOS Figure 7: exclude from reproduction until the prior-study photographs are",
        "  independently cleared.",
        "- Research Square Figure 1: exclude from reproduction because it contains",
        "  BioRender-labelled material with separate reuse conditions.",
        "",
        "This is a conservative engineering disposition, not legal or rights-holder advice.",
        "",
    ]
    args.summary.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Recorded conservative dispositions for {len(output_rows)} figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
