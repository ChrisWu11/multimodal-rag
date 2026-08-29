from __future__ import annotations

import math
import random
import re
from collections.abc import Iterable, Sequence
from statistics import fmean
from typing import Any


def clean(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def normalise_doi(value: Any) -> str:
    doi = clean(value)
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        doi = doi.removeprefix(prefix)
    return doi.rstrip(".,;")


def result_matches_gold(result: dict[str, Any], gold: dict[str, Any]) -> bool:
    if gold.get("chunk_id") and clean(result.get("chunk_id")) == clean(gold.get("chunk_id")):
        return True
    if gold.get("doc_id") and clean(result.get("doc_id")) == clean(gold.get("doc_id")):
        return True
    if gold.get("doi") and normalise_doi(result.get("doi")) == normalise_doi(gold.get("doi")):
        return True
    title_contains = clean(gold.get("title_contains"))
    return bool(title_contains and title_contains in clean(result.get("title")))


def first_relevant_rank(
    results: Sequence[dict[str, Any]],
    gold_sources: Sequence[dict[str, Any]],
) -> int | None:
    for rank, result in enumerate(results, start=1):
        if any(result_matches_gold(result, gold) for gold in gold_sources):
            return rank
    return None


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def ndcg_at_k(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def summarise_ranks(ranks: Iterable[int | None], k_values: Sequence[int]) -> dict[str, float]:
    rank_list = list(ranks)
    count = len(rank_list)
    if not count:
        return {"judged_count": 0, "mrr": 0.0}
    summary: dict[str, float] = {
        "judged_count": count,
        "mrr": round(sum(reciprocal_rank(rank) for rank in rank_list) / count, 4),
    }
    for k in k_values:
        summary[f"recall_at_{k}"] = round(
            sum(rank is not None and rank <= k for rank in rank_list) / count,
            4,
        )
        summary[f"ndcg_at_{k}"] = round(
            sum(ndcg_at_k(rank, k) for rank in rank_list) / count,
            4,
        )
    return summary


def citation_indices(answer: str) -> list[int]:
    return [int(value) for value in re.findall(r"\[(\d+)\]", answer)]


def citation_structure(answer: str, evidence_count: int) -> dict[str, Any]:
    indices = citation_indices(answer)
    invalid = sorted({index for index in indices if index < 1 or index > evidence_count})
    valid = [index for index in indices if 1 <= index <= evidence_count]
    return {
        "citation_count": len(indices),
        "unique_valid_citations": len(set(valid)),
        "invalid_citations": invalid,
        "has_sources_section": "sources used" in answer.lower(),
    }


def percentile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated percentile for a sorted numeric sample."""
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 20_260_723,
) -> dict[str, float | int]:
    """Estimate a percentile bootstrap confidence interval for a sample mean."""
    sample = [float(value) for value in values]
    if not sample:
        raise ValueError("bootstrap_mean_ci requires at least one value")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if resamples < 1:
        raise ValueError("resamples must be positive")

    rng = random.Random(seed)
    count = len(sample)
    estimates = [
        fmean(sample[rng.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    ]
    alpha = (1 - confidence) / 2
    return {
        "estimate": fmean(sample),
        "ci_low": percentile(estimates, alpha),
        "ci_high": percentile(estimates, 1 - alpha),
        "confidence": confidence,
        "resamples": resamples,
        "sample_size": count,
    }


def bootstrap_ratio_ci(
    numerators: Sequence[float],
    denominators: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 20_260_723,
) -> dict[str, float | int] | None:
    """Bootstrap a ratio of paired totals, such as unsupported claims / claims."""
    if len(numerators) != len(denominators):
        raise ValueError("numerators and denominators must have the same length")
    pairs = [
        (float(numerator), float(denominator))
        for numerator, denominator in zip(numerators, denominators)
    ]
    if not pairs or sum(denominator for _, denominator in pairs) == 0:
        return None

    rng = random.Random(seed)
    count = len(pairs)
    estimates: list[float] = []
    for _ in range(resamples):
        sampled = [pairs[rng.randrange(count)] for _ in range(count)]
        denominator = sum(item[1] for item in sampled)
        if denominator:
            estimates.append(sum(item[0] for item in sampled) / denominator)
    alpha = (1 - confidence) / 2
    return {
        "estimate": sum(item[0] for item in pairs) / sum(item[1] for item in pairs),
        "ci_low": percentile(estimates, alpha),
        "ci_high": percentile(estimates, 1 - alpha),
        "confidence": confidence,
        "resamples": len(estimates),
        "sample_size": count,
    }


def paired_randomisation_test(
    left: Sequence[float],
    right: Sequence[float],
    *,
    resamples: int = 50_000,
    seed: int = 20_260_723,
) -> dict[str, float | int]:
    """Two-sided paired randomisation test using random sign flips."""
    if len(left) != len(right):
        raise ValueError("paired samples must have the same length")
    if not left:
        raise ValueError("paired_randomisation_test requires at least one pair")
    differences = [
        float(left_value) - float(right_value)
        for left_value, right_value in zip(left, right)
    ]
    observed = fmean(differences)
    if all(difference == 0 for difference in differences):
        return {
            "mean_difference": 0.0,
            "p_value": 1.0,
            "resamples": resamples,
            "sample_size": len(differences),
        }

    rng = random.Random(seed)
    extreme = 0
    for _ in range(resamples):
        permuted = fmean(
            difference if rng.random() < 0.5 else -difference
            for difference in differences
        )
        if abs(permuted) >= abs(observed) - 1e-15:
            extreme += 1
    return {
        "mean_difference": observed,
        "p_value": (extreme + 1) / (resamples + 1),
        "resamples": resamples,
        "sample_size": len(differences),
    }


def quadratic_weighted_kappa(
    left: Sequence[int],
    right: Sequence[int],
    *,
    minimum: int = 0,
    maximum: int = 2,
) -> float:
    """Compute quadratic-weighted Cohen's kappa for two ordinal ratings."""
    if len(left) != len(right):
        raise ValueError("ratings must have the same length")
    if not left:
        raise ValueError("quadratic_weighted_kappa requires at least one pair")
    categories = maximum - minimum + 1
    if categories < 2:
        raise ValueError("at least two categories are required")
    if any(value < minimum or value > maximum for value in (*left, *right)):
        raise ValueError("rating outside the configured range")

    observed = [[0.0] * categories for _ in range(categories)]
    left_counts = [0.0] * categories
    right_counts = [0.0] * categories
    for left_value, right_value in zip(left, right):
        i = left_value - minimum
        j = right_value - minimum
        observed[i][j] += 1
        left_counts[i] += 1
        right_counts[j] += 1

    count = float(len(left))
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    scale = float((categories - 1) ** 2)
    for i in range(categories):
        for j in range(categories):
            weight = ((i - j) ** 2) / scale
            observed_disagreement += weight * observed[i][j] / count
            expected_disagreement += weight * (
                left_counts[i] * right_counts[j] / (count * count)
            )
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else 0.0
    return 1 - observed_disagreement / expected_disagreement
