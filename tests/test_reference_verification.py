from scripts.verify_report_references import author_checks, given_names_match


def test_given_name_comparison_allows_initials_but_not_different_names() -> None:
    assert given_names_match("C. T. W.", "Chrit T. W.")
    assert given_names_match("Philip S.", "Philip")
    assert not given_names_match("Antoine", "Adrien")


def test_author_check_detects_same_initial_wrong_given_name() -> None:
    fields = {"author": "Rohfritsch, Antoine and Barrere, Vincent"}
    registry = {
        "author_parts": [
            {"given": "Adrien", "family": "Rohfritsch"},
            {"given": "Victor", "family": "Barrere"},
        ]
    }

    discrepancies = author_checks(fields, registry)

    assert len(discrepancies) == 2
    assert "Antoine Rohfritsch" in discrepancies[0]
    assert "Vincent Barrere" in discrepancies[1]
