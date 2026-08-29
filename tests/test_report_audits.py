from __future__ import annotations

import csv
from pathlib import Path

from scripts.finalise_image_rights_disposition import FLAGGED_ACTIONS
from scripts.review_report_claim_citations import REVIEWS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_claim_audit_covers_every_generated_human_review_row() -> None:
    ai_rows = read_rows(
        PROJECT_ROOT
        / "evaluation"
        / "manual-review"
        / "04-claim-citation-ai-review.csv"
    )
    human_rows = read_rows(
        PROJECT_ROOT
        / "evaluation"
        / "manual-review"
        / "04-claim-citation-support-review.csv"
    )

    assert len(REVIEWS) == len(ai_rows) == len(human_rows) == 17
    assert {row["claim_id"] for row in ai_rows} == {
        f"c{index:02d}" for index in range(1, 18)
    }
    assert all(row["final_assessment"].startswith("supported") for row in ai_rows)
    assert all(row["independent_human_review"] == "not_performed" for row in ai_rows)
    assert all(row["review_status"] == "pending" for row in human_rows)


def test_image_disposition_never_releases_source_figures() -> None:
    rows = read_rows(
        PROJECT_ROOT
        / "evaluation"
        / "manual-review"
        / "03-image-rights-ai-disposition.csv"
    )

    assert len(rows) == 12
    assert all(row["source_image_in_final_report"] == "no" for row in rows)
    assert all(row["source_image_in_public_package"] == "no" for row in rows)
    assert all(
        row["independent_rights_holder_confirmation"] == "not_performed"
        for row in rows
    )
    assert set(FLAGGED_ACTIONS).issubset({row["candidate_id"] for row in rows})
    assert {
        row["public_disposition"]
        for row in rows
        if row["candidate_id"] in FLAGGED_ACTIONS
    } == {action["public_disposition"] for action in FLAGGED_ACTIONS.values()}
