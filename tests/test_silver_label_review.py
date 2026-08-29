import csv
from pathlib import Path

from scripts.review_silver_labels import APPROVED_NOTES, CORRECTIONS


ROOT = Path(__file__).resolve().parents[1]
REVIEW_CSV = ROOT / "evaluation" / "manual-review" / "01-silver-question-ai-review.csv"


def test_review_map_covers_all_labels_once() -> None:
    approved = set(APPROVED_NOTES)
    corrected = set(CORRECTIONS)
    expected = {f"q{index:03d}" for index in range(1, 51)}

    assert not approved & corrected
    assert approved | corrected == expected
    assert len(approved) == 32
    assert len(corrected) == 18


def test_generated_review_audit_is_complete_and_non_expert() -> None:
    with REVIEW_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 50
    assert len({row["id"] for row in rows}) == 50
    assert {row["decision"] for row in rows} == {"approved", "corrected"}
    assert sum(row["decision"] == "approved" for row in rows) == 32
    assert sum(row["decision"] == "corrected" for row in rows) == 18
    assert sum(row["confidence"] == "high" for row in rows) == 47
    assert sum(row["confidence"] == "medium" for row in rows) == 3
    assert all("non-clinical" in row["reviewer_type"] for row in rows)
    assert all(row["corrected_gold_chunk_id"] for row in rows)
    assert all(row["corrected_gold_answer"] for row in rows)


def test_every_doi_record_has_a_matching_registry_title() -> None:
    with REVIEW_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    doi_rows = [row for row in rows if row["corrected_gold_source_doi"]]
    repository_rows = [row for row in rows if not row["corrected_gold_source_doi"]]

    assert len(doi_rows) == 46
    assert all(row["doi_registry_verified"] == "True" for row in doi_rows)
    assert all(row["registered_title_match"] == "True" for row in doi_rows)
    assert len(repository_rows) == 4
    assert all(row["doi_registry_verified"] == "not_applicable" for row in repository_rows)
