from evaluation.metrics import (
    bootstrap_mean_ci,
    bootstrap_ratio_ci,
    citation_structure,
    first_relevant_rank,
    ndcg_at_k,
    normalise_doi,
    paired_randomisation_test,
    quadratic_weighted_kappa,
    summarise_ranks,
)


def test_retrieval_metrics_with_single_relevant_source() -> None:
    results = [
        {"chunk_id": "c1", "doc_id": "d1", "doi": "10.1/a"},
        {"chunk_id": "c2", "doc_id": "d2", "doi": "10.1/b"},
    ]
    rank = first_relevant_rank(results, [{"doc_id": "d2"}])
    assert rank == 2
    assert round(ndcg_at_k(rank, 5), 4) == 0.6309
    assert summarise_ranks([1, 2, None], [1, 5]) == {
        "judged_count": 3,
        "mrr": 0.5,
        "recall_at_1": 0.3333,
        "ndcg_at_1": 0.3333,
        "recall_at_5": 0.6667,
        "ndcg_at_5": 0.5436,
    }


def test_doi_and_citation_validation() -> None:
    assert normalise_doi("https://doi.org/10.1000/Example.") == "10.1000/example"
    assert citation_structure("Claim [1]. Another [3]. Sources used", 2) == {
        "citation_count": 2,
        "unique_valid_citations": 1,
        "invalid_citations": [3],
        "has_sources_section": True,
    }


def test_bootstrap_intervals_are_deterministic_and_contain_estimate() -> None:
    interval = bootstrap_mean_ci([0, 0, 1, 1], resamples=500, seed=7)
    assert interval["estimate"] == 0.5
    assert interval["ci_low"] <= interval["estimate"] <= interval["ci_high"]
    assert interval == bootstrap_mean_ci([0, 0, 1, 1], resamples=500, seed=7)

    ratio = bootstrap_ratio_ci([0, 1, 0], [2, 2, 2], resamples=500, seed=7)
    assert ratio is not None
    assert round(float(ratio["estimate"]), 4) == 0.1667
    assert bootstrap_ratio_ci([0], [0], resamples=10) is None


def test_paired_significance_and_ordinal_agreement() -> None:
    unchanged = paired_randomisation_test([1, 2, 3], [1, 2, 3], resamples=100)
    assert unchanged["mean_difference"] == 0
    assert unchanged["p_value"] == 1

    comparison = paired_randomisation_test(
        [1] * 12,
        [0] * 12,
        resamples=2_000,
        seed=3,
    )
    assert comparison["mean_difference"] == 1
    assert comparison["p_value"] < 0.01

    assert quadratic_weighted_kappa([0, 1, 2], [0, 1, 2]) == 1
    assert quadratic_weighted_kappa([0, 0, 2, 2], [0, 1, 1, 2]) > 0
