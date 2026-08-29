#!/usr/bin/env python3
"""Add confidence intervals, paired tests, agreement and error analysis to report runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import fmean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import (  # noqa: E402
    bootstrap_mean_ci,
    bootstrap_ratio_ci,
    ndcg_at_k,
    paired_randomisation_test,
    quadratic_weighted_kappa,
    reciprocal_rank,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "final-report-evaluation"
SEED = 20_260_723
ORDINAL_METRICS = ("correctness", "faithfulness", "completeness")
IMAGE_METRICS = ("caption_coverage", "citation_quality", "response_structure")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


def create_run_dir(output_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_root / f"{stamp}-statistical-analysis"
    suffix = 1
    while path.exists():
        path = output_root / f"{stamp}-statistical-analysis-{suffix}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def rounded_interval(
    values: list[float],
    *,
    resamples: int,
    seed_offset: int,
) -> dict[str, Any]:
    result = bootstrap_mean_ci(
        values,
        resamples=resamples,
        seed=SEED + seed_offset,
    )
    return {
        **result,
        "estimate": round(float(result["estimate"]), 4),
        "ci_low": round(float(result["ci_low"]), 4),
        "ci_high": round(float(result["ci_high"]), 4),
    }


def rounded_ratio_interval(
    numerators: list[float],
    denominators: list[float],
    *,
    resamples: int,
    seed_offset: int,
) -> dict[str, Any] | None:
    result = bootstrap_ratio_ci(
        numerators,
        denominators,
        resamples=resamples,
        seed=SEED + seed_offset,
    )
    if result is None:
        return None
    return {
        **result,
        "estimate": round(float(result["estimate"]), 4),
        "ci_low": round(float(result["ci_low"]), 4),
        "ci_high": round(float(result["ci_high"]), 4),
    }


def rounded_paired_test(
    left: list[float],
    right: list[float],
    *,
    resamples: int,
    seed_offset: int,
) -> dict[str, Any]:
    result = paired_randomisation_test(
        left,
        right,
        resamples=resamples,
        seed=SEED + seed_offset,
    )
    return {
        **result,
        "mean_difference": round(float(result["mean_difference"]), 4),
        "p_value": round(float(result["p_value"]), 6),
    }


def retrieval_analysis(
    details_path: Path,
    *,
    bootstrap_resamples: int,
    permutation_resamples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = load(details_path)
    by_mode: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_mode[str(row["mode"])][str(row["id"])] = row

    metric_functions = {
        "mrr": lambda rank: reciprocal_rank(rank),
        "recall_at_5": lambda rank: float(rank is not None and rank <= 5),
        "recall_at_10": lambda rank: float(rank is not None and rank <= 10),
        "recall_at_20": lambda rank: float(rank is not None and rank <= 20),
        "ndcg_at_5": lambda rank: ndcg_at_k(rank, 5),
        "ndcg_at_10": lambda rank: ndcg_at_k(rank, 10),
    }
    modes: dict[str, Any] = {}
    value_cache: dict[tuple[str, str], dict[str, float]] = {}
    seed_offset = 0
    for mode, mode_rows in sorted(by_mode.items()):
        modes[mode] = {}
        for metric, function in metric_functions.items():
            values_by_id = {
                row_id: function(row.get("hit_rank"))
                for row_id, row in mode_rows.items()
            }
            value_cache[(mode, metric)] = values_by_id
            modes[mode][metric] = rounded_interval(
                list(values_by_id.values()),
                resamples=bootstrap_resamples,
                seed_offset=seed_offset,
            )
            seed_offset += 1

    comparisons = []
    requested = [
        ("st_hybrid_rrf", "fts_bm25", "Hybrid RRF versus lexical"),
        ("st_hybrid_rrf", "st_dense", "Hybrid RRF versus dense"),
        (
            "rerank_hybrid_rrf_top30",
            "st_hybrid_rrf",
            "Hybrid reranking versus Hybrid RRF",
        ),
        ("rerank_fts_top30", "fts_bm25", "Lexical reranking versus lexical"),
    ]
    for left_mode, right_mode, label in requested:
        if left_mode not in by_mode or right_mode not in by_mode:
            continue
        shared_ids = sorted(set(by_mode[left_mode]) & set(by_mode[right_mode]))
        for metric in ("mrr", "recall_at_5", "ndcg_at_10"):
            left = [value_cache[(left_mode, metric)][row_id] for row_id in shared_ids]
            right = [value_cache[(right_mode, metric)][row_id] for row_id in shared_ids]
            result = rounded_paired_test(
                left,
                right,
                resamples=permutation_resamples,
                seed_offset=seed_offset,
            )
            seed_offset += 1
            comparisons.append(
                {
                    "comparison": label,
                    "left_mode": left_mode,
                    "right_mode": right_mode,
                    "metric": metric,
                    **result,
                    "interpretation": (
                        "statistically_detectable"
                        if float(result["p_value"]) < 0.05
                        else "not_detected_at_0.05"
                    ),
                }
            )

    diagnostic_rows: list[dict[str, Any]] = []
    core_modes = {
        "lexical": "fts_bm25",
        "dense": "st_dense",
        "hybrid": "st_hybrid_rrf",
        "reranked": "rerank_hybrid_rrf_top30",
    }
    available = all(mode in by_mode for mode in core_modes.values())
    if available:
        ids = sorted(set.intersection(*(set(by_mode[mode]) for mode in core_modes.values())))
        for row_id in ids:
            source_rows = {
                label: by_mode[mode][row_id] for label, mode in core_modes.items()
            }
            ranks = {label: row.get("hit_rank") for label, row in source_rows.items()}
            hybrid_rank = ranks["hybrid"]
            reranked_rank = ranks["reranked"]
            if reranked_rank is None or reranked_rank > 5:
                if hybrid_rank is None or hybrid_rank > 30:
                    category = "candidate_omission"
                elif hybrid_rank <= 5:
                    category = "reranker_demotion"
                else:
                    category = "relevant_evidence_below_context_cutoff"
            elif (
                (ranks["lexical"] is None or ranks["lexical"] > 5)
                and (ranks["dense"] is None or ranks["dense"] > 5)
                and hybrid_rank is not None
                and hybrid_rank <= 5
            ):
                category = "fusion_rescue"
            elif hybrid_rank is not None and hybrid_rank > 5 and reranked_rank <= 5:
                category = "reranker_promotion"
            elif ranks["lexical"] is not None and ranks["lexical"] <= 5 and (
                ranks["dense"] is None or ranks["dense"] > 5
            ):
                category = "lexical_strength"
            elif ranks["dense"] is not None and ranks["dense"] <= 5 and (
                ranks["lexical"] is None or ranks["lexical"] > 5
            ):
                category = "semantic_strength"
            else:
                category = "shared_success"
            diagnostic_rows.append(
                {
                    "stage": "retrieval",
                    "id": row_id,
                    "question": source_rows["hybrid"].get("question"),
                    "category": category,
                    "lexical_rank": ranks["lexical"],
                    "dense_rank": ranks["dense"],
                    "hybrid_rank": hybrid_rank,
                    "reranked_rank": reranked_rank,
                    "requires_human_label_review": True,
                }
            )

    return (
        {
            "question_count": len(next(iter(by_mode.values()))) if by_mode else 0,
            "confidence_intervals": modes,
            "paired_randomisation_tests": comparisons,
            "diagnostic_counts": dict(
                sorted(Counter(row["category"] for row in diagnostic_rows).items())
            ),
        },
        diagnostic_rows,
    )


def rank_sensitivity_analysis(
    details_path: Path,
    *,
    kind: str,
    bootstrap_resamples: int,
    permutation_resamples: int,
) -> dict[str, Any]:
    rows = load(details_path)
    if kind == "metadata":
        def group_label(row: dict[str, Any]) -> str:
            return f"{row['configuration']}:{row['mode']}"

        requested = [
            (
                "title_section_text:hybrid_rrf",
                "text_only:hybrid_rrf",
                "Production metadata versus text-only hybrid",
            ),
            (
                "title_section_text:dense",
                "text_only:dense",
                "Production metadata versus text-only dense",
            ),
            (
                "title_section_text:hybrid_rrf",
                "title_text:hybrid_rrf",
                "Title and section versus title-only hybrid",
            ),
        ]
    elif kind == "reranker_pool":
        def group_label(row: dict[str, Any]) -> str:
            return f"top_{int(row['pool_size'])}"

        requested = [
            ("top_20", "top_10", "Top-20 versus top-10 candidate pool"),
            ("top_30", "top_20", "Top-30 versus top-20 candidate pool"),
            ("top_50", "top_20", "Top-50 versus top-20 candidate pool"),
        ]
    elif kind == "chunk":
        def group_label(row: dict[str, Any]) -> str:
            return str(row["configuration"])

        requested = [
            ("w350_o50", "w650_o90", "350/50 versus production 650/90"),
            ("w650_o0", "w650_o90", "No overlap versus production 650/90"),
            ("w900_o150", "w650_o90", "900/150 versus production 650/90"),
        ]
    else:
        raise ValueError(f"Unsupported sensitivity analysis kind: {kind}")

    grouped: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[group_label(row)][str(row["id"])] = row
    metric_functions = {
        "mrr": lambda rank: reciprocal_rank(rank),
        "recall_at_5": lambda rank: float(rank is not None and rank <= 5),
        "recall_at_10": lambda rank: float(rank is not None and rank <= 10),
        "recall_at_20": lambda rank: float(rank is not None and rank <= 20),
        "ndcg_at_5": lambda rank: ndcg_at_k(rank, 5),
        "ndcg_at_10": lambda rank: ndcg_at_k(rank, 10),
    }
    confidence_intervals: dict[str, Any] = {}
    value_cache: dict[tuple[str, str], dict[str, float]] = {}
    seed_offset = {"metadata": 500, "reranker_pool": 700, "chunk": 900}[kind]
    for group, group_rows in sorted(grouped.items()):
        confidence_intervals[group] = {}
        for metric, function in metric_functions.items():
            values = {
                row_id: function(row.get("hit_rank"))
                for row_id, row in group_rows.items()
            }
            value_cache[(group, metric)] = values
            confidence_intervals[group][metric] = rounded_interval(
                list(values.values()),
                resamples=bootstrap_resamples,
                seed_offset=seed_offset,
            )
            seed_offset += 1

    comparisons = []
    for left_group, right_group, label in requested:
        if left_group not in grouped or right_group not in grouped:
            continue
        shared_ids = sorted(set(grouped[left_group]) & set(grouped[right_group]))
        for metric in ("mrr", "recall_at_5", "ndcg_at_10"):
            result = rounded_paired_test(
                [value_cache[(left_group, metric)][row_id] for row_id in shared_ids],
                [value_cache[(right_group, metric)][row_id] for row_id in shared_ids],
                resamples=permutation_resamples,
                seed_offset=seed_offset,
            )
            seed_offset += 1
            comparisons.append(
                {
                    "comparison": label,
                    "left_mode": left_group,
                    "right_mode": right_group,
                    "metric": metric,
                    **result,
                    "interpretation": (
                        "statistically_detectable"
                        if float(result["p_value"]) < 0.05
                        else "not_detected_at_0.05"
                    ),
                }
            )

    latency = {}
    for group, group_rows in grouped.items():
        values = [float(row["latency_ms"]) for row in group_rows.values()]
        latency[group] = {
            "mean_latency_ms": round(fmean(values), 2),
            "question_count": len(values),
        }
    return {
        "kind": kind,
        "confidence_intervals": confidence_intervals,
        "paired_randomisation_tests": comparisons,
        "latency": dict(sorted(latency.items())),
    }


def generation_analysis(
    answers_path: Path,
    judgments_path: Path,
    *,
    bootstrap_resamples: int,
    permutation_resamples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    answers = load(answers_path)
    judgments = load(judgments_path)
    answer_by_key = {
        (str(row["id"]), str(row["mode"])): row
        for row in answers
    }
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    raw_grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in judgments:
        key = (str(row["id"]), str(row["mode"]))
        grouped[key].append(row["judge"])
        raw_grouped[key].append(row)

    aggregate: dict[tuple[str, str], dict[str, Any]] = {}
    for key, judge_rows in grouped.items():
        answer = answer_by_key[key]
        aggregate[key] = {
            "id": key[0],
            "mode": key[1],
            "question": answer["question"],
            "answerable": bool(answer["answerable"]),
            "invalid_citations": answer.get("invalid_citations") or [],
            **{
                metric: fmean(float(row.get(metric) or 0) for row in judge_rows)
                for metric in ORDINAL_METRICS
            },
            "factual_claims": fmean(
                float(row.get("factual_claims") or 0) for row in judge_rows
            ),
            "unsupported_claims": fmean(
                float(row.get("unsupported_claims") or 0) for row in judge_rows
            ),
            "claims_with_citations": fmean(
                float(row.get("claims_with_citations") or 0) for row in judge_rows
            ),
            "supported_cited_claims": fmean(
                float(row.get("supported_cited_claims") or 0) for row in judge_rows
            ),
            "abstained": fmean(float(bool(row.get("abstained"))) for row in judge_rows),
            "judge_reasons": " | ".join(
                dict.fromkeys(str(row.get("brief_reason") or "") for row in judge_rows)
            ),
        }

    modes = sorted({key[1] for key in aggregate})
    summaries: dict[str, Any] = {}
    seed_offset = 100
    for mode in modes:
        mode_rows = [row for key, row in aggregate.items() if key[1] == mode]
        answerable = [row for row in mode_rows if row["answerable"]]
        unanswerable = [row for row in mode_rows if not row["answerable"]]
        summary: dict[str, Any] = {
            "answer_count": len(mode_rows),
            "answerable_count": len(answerable),
            "unanswerable_count": len(unanswerable),
        }
        for metric in ORDINAL_METRICS:
            summary[metric] = rounded_interval(
                [float(row[metric]) for row in answerable],
                resamples=bootstrap_resamples,
                seed_offset=seed_offset,
            )
            seed_offset += 1
        summary["unsupported_claim_rate"] = rounded_ratio_interval(
            [float(row["unsupported_claims"]) for row in mode_rows],
            [float(row["factual_claims"]) for row in mode_rows],
            resamples=bootstrap_resamples,
            seed_offset=seed_offset,
        )
        seed_offset += 1
        summary["citation_precision"] = (
            None
            if mode == "llm_only"
            else rounded_ratio_interval(
                [float(row["supported_cited_claims"]) for row in answerable],
                [float(row["claims_with_citations"]) for row in answerable],
                resamples=bootstrap_resamples,
                seed_offset=seed_offset,
            )
        )
        seed_offset += 1
        summary["abstention_accuracy"] = rounded_interval(
            [float(row["abstained"]) for row in unanswerable],
            resamples=bootstrap_resamples,
            seed_offset=seed_offset,
        )
        seed_offset += 1
        summary["invalid_citation_count"] = sum(
            len(row["invalid_citations"]) for row in mode_rows
        )
        summaries[mode] = summary

    comparisons = []
    for left_mode, right_mode, label in [
        ("dense_rag", "llm_only", "Dense RAG versus LLM only"),
        (
            "hybrid_reranked_rag",
            "dense_rag",
            "Hybrid reranked RAG versus dense RAG",
        ),
        (
            "hybrid_reranked_rag",
            "llm_only",
            "Hybrid reranked RAG versus LLM only",
        ),
    ]:
        ids = sorted(
            {
                key[0]
                for key, row in aggregate.items()
                if key[1] == left_mode and row["answerable"]
            }
            & {
                key[0]
                for key, row in aggregate.items()
                if key[1] == right_mode and row["answerable"]
            }
        )
        for metric in ORDINAL_METRICS:
            result = rounded_paired_test(
                [float(aggregate[(row_id, left_mode)][metric]) for row_id in ids],
                [float(aggregate[(row_id, right_mode)][metric]) for row_id in ids],
                resamples=permutation_resamples,
                seed_offset=seed_offset,
            )
            seed_offset += 1
            comparisons.append(
                {
                    "comparison": label,
                    "left_mode": left_mode,
                    "right_mode": right_mode,
                    "metric": metric,
                    **result,
                    "interpretation": (
                        "statistically_detectable"
                        if float(result["p_value"]) < 0.05
                        else "not_detected_at_0.05"
                    ),
                }
            )

    agreement_rows = []
    for mode in modes:
        for metric in ORDINAL_METRICS:
            pairs = []
            for key, raw_rows in raw_grouped.items():
                if key[1] != mode or len(raw_rows) < 2:
                    continue
                ordered = sorted(raw_rows, key=lambda row: int(row["repeat"]))
                pairs.append(
                    (
                        int(ordered[0]["judge"].get(metric) or 0),
                        int(ordered[1]["judge"].get(metric) or 0),
                    )
                )
            if not pairs:
                continue
            left, right = zip(*pairs)
            agreement_rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "pair_count": len(pairs),
                    "exact_agreement": round(
                        sum(a == b for a, b in pairs) / len(pairs),
                        4,
                    ),
                    "quadratic_weighted_kappa": round(
                        quadratic_weighted_kappa(left, right),
                        4,
                    ),
                }
            )
        abstention_pairs = []
        for key, raw_rows in raw_grouped.items():
            answer = answer_by_key.get(key)
            if key[1] != mode or not answer or answer["answerable"] or len(raw_rows) < 2:
                continue
            ordered = sorted(raw_rows, key=lambda row: int(row["repeat"]))
            abstention_pairs.append(
                (
                    int(bool(ordered[0]["judge"].get("abstained"))),
                    int(bool(ordered[1]["judge"].get("abstained"))),
                )
            )
        if abstention_pairs:
            agreement_rows.append(
                {
                    "mode": mode,
                    "metric": "abstention",
                    "pair_count": len(abstention_pairs),
                    "exact_agreement": round(
                        sum(a == b for a, b in abstention_pairs)
                        / len(abstention_pairs),
                        4,
                    ),
                    "quadratic_weighted_kappa": round(
                        quadratic_weighted_kappa(
                            [pair[0] for pair in abstention_pairs],
                            [pair[1] for pair in abstention_pairs],
                            maximum=1,
                        ),
                        4,
                    ),
                }
            )

    error_rows: list[dict[str, Any]] = []
    for row in aggregate.values():
        categories = []
        citation_denominator = float(row["claims_with_citations"])
        citation_precision = (
            float(row["supported_cited_claims"]) / citation_denominator
            if citation_denominator
            else None
        )
        if not row["answerable"] and float(row["abstained"]) < 1:
            categories.append("failed_abstention")
        if row["invalid_citations"]:
            categories.append("invalid_citation_index")
        if float(row["unsupported_claims"]) > 0:
            categories.append("unsupported_claim")
        if row["answerable"] and float(row["faithfulness"]) < 2:
            categories.append("faithfulness_below_maximum")
        if row["answerable"] and float(row["correctness"]) < 2:
            categories.append("correctness_below_maximum")
        if (
            row["mode"] != "llm_only"
            and citation_precision is not None
            and citation_precision < 1
        ):
            categories.append("citation_support_gap")
        for category in categories:
            error_rows.append(
                {
                    "stage": "generation",
                    "id": row["id"],
                    "mode": row["mode"],
                    "question": row["question"],
                    "category": category,
                    "correctness": round(float(row["correctness"]), 3),
                    "faithfulness": round(float(row["faithfulness"]), 3),
                    "completeness": round(float(row["completeness"]), 3),
                    "unsupported_claims": round(float(row["unsupported_claims"]), 3),
                    "citation_precision": (
                        round(citation_precision, 3)
                        if citation_precision is not None
                        else None
                    ),
                    "judge_reason": row["judge_reasons"],
                    "requires_human_scientific_review": True,
                }
            )

    return (
        {
            "quality_scope": "Answerable questions only for correctness, faithfulness and completeness.",
            "safety_scope": "All cases for unsupported claims; unanswerable cases for abstention.",
            "citation_scope": "Citation precision is not applicable to the no-evidence baseline.",
            "judge_models": dict(
                sorted(Counter(str(row.get("judge_model")) for row in judgments).items())
            ),
            "summaries": summaries,
            "paired_randomisation_tests": comparisons,
            "judge_repeat_agreement": agreement_rows,
            "diagnostic_counts": dict(
                sorted(Counter(row["category"] for row in error_rows).items())
            ),
        },
        error_rows,
    )


def image_analysis(
    answers_path: Path,
    judgments_path: Path,
    retrieval_path: Path,
    *,
    bootstrap_resamples: int,
    permutation_resamples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    answers = load(answers_path)
    judgments = load(judgments_path)
    retrieval_rows = load(retrieval_path)
    answers_by_key = {
        (str(row["candidate_id"]), str(row["mode"])): row
        for row in answers
    }
    retrieval_by_key = {
        (str(row["candidate_id"]), str(row["mode"])): row
        for row in retrieval_rows
    }
    judges_by_key = {
        (str(row["candidate_id"]), str(row["mode"])): row["judge"]
        for row in judgments
    }
    modes = sorted({key[1] for key in judges_by_key})
    summaries: dict[str, Any] = {}
    seed_offset = 300
    for mode in modes:
        keys = sorted(key for key in judges_by_key if key[1] == mode)
        summary: dict[str, Any] = {"image_count": len(keys)}
        for metric in IMAGE_METRICS:
            summary[metric] = rounded_interval(
                [float(judges_by_key[key].get(metric) or 0) for key in keys],
                resamples=bootstrap_resamples,
                seed_offset=seed_offset,
            )
            seed_offset += 1
        summary["source_recall_at_5"] = rounded_interval(
            [float(bool(retrieval_by_key[key]["source_hit_at_5"])) for key in keys],
            resamples=bootstrap_resamples,
            seed_offset=seed_offset,
        )
        seed_offset += 1
        summary["unsupported_visual_inference_rate"] = rounded_interval(
            [
                float((judges_by_key[key].get("unsupported_visual_inferences") or 0) > 0)
                for key in keys
            ],
            resamples=bootstrap_resamples,
            seed_offset=seed_offset,
        )
        seed_offset += 1
        summary["visual_contradiction_count"] = sum(
            int(judges_by_key[key].get("visual_contradictions") or 0)
            for key in keys
        )
        summary["invalid_citation_count"] = sum(
            len(answers_by_key[key].get("invalid_citations") or [])
            for key in keys
        )
        summaries[mode] = summary

    comparisons = []
    for left_mode, right_mode, label in [
        ("vision_summary", "text_only", "Vision summary versus text only"),
        ("vision_summary", "basic_metadata", "Vision summary versus basic metadata"),
    ]:
        ids = sorted(
            {key[0] for key in judges_by_key if key[1] == left_mode}
            & {key[0] for key in judges_by_key if key[1] == right_mode}
        )
        for metric in (*IMAGE_METRICS, "source_recall_at_5"):
            if metric == "source_recall_at_5":
                left = [
                    float(bool(retrieval_by_key[(row_id, left_mode)]["source_hit_at_5"]))
                    for row_id in ids
                ]
                right = [
                    float(bool(retrieval_by_key[(row_id, right_mode)]["source_hit_at_5"]))
                    for row_id in ids
                ]
            else:
                left = [
                    float(judges_by_key[(row_id, left_mode)].get(metric) or 0)
                    for row_id in ids
                ]
                right = [
                    float(judges_by_key[(row_id, right_mode)].get(metric) or 0)
                    for row_id in ids
                ]
            result = rounded_paired_test(
                left,
                right,
                resamples=permutation_resamples,
                seed_offset=seed_offset,
            )
            seed_offset += 1
            comparisons.append(
                {
                    "comparison": label,
                    "left_mode": left_mode,
                    "right_mode": right_mode,
                    "metric": metric,
                    **result,
                    "interpretation": (
                        "statistically_detectable"
                        if float(result["p_value"]) < 0.05
                        else "not_detected_at_0.05"
                    ),
                }
            )

    error_rows: list[dict[str, Any]] = []
    for key, judge in judges_by_key.items():
        candidate_id, mode = key
        retrieval = retrieval_by_key[key]
        answer = answers_by_key[key]
        categories = []
        if not retrieval["source_hit_at_5"]:
            categories.append("source_retrieval_failure")
        if int(judge.get("unsupported_visual_inferences") or 0):
            categories.append("unsupported_visual_inference")
        if int(judge.get("visual_contradictions") or 0):
            categories.append("visual_contradiction")
        if float(judge.get("caption_coverage") or 0) < 2:
            categories.append("caption_coverage_below_maximum")
        if float(judge.get("citation_quality") or 0) < 2:
            categories.append("citation_quality_below_maximum")
        if answer.get("invalid_citations"):
            categories.append("invalid_citation_index")
        for category in categories:
            error_rows.append(
                {
                    "stage": "image_qa",
                    "id": candidate_id,
                    "mode": mode,
                    "question": retrieval.get("question"),
                    "category": category,
                    "source_rank": retrieval.get("source_rank"),
                    "caption_coverage": judge.get("caption_coverage"),
                    "citation_quality": judge.get("citation_quality"),
                    "unsupported_visual_inferences": judge.get(
                        "unsupported_visual_inferences"
                    ),
                    "judge_reason": judge.get("brief_reason"),
                    "requires_human_scientific_review": True,
                }
            )

    return (
        {
            "summaries": summaries,
            "paired_randomisation_tests": comparisons,
            "diagnostic_counts": dict(
                sorted(Counter(row["category"] for row in error_rows).items())
            ),
            "scope_warning": (
                "Caption-derived exploratory questions are not independent clinical labels."
            ),
        },
        error_rows,
    )


def flatten_intervals(section: str, summaries: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for mode, metrics in summaries.items():
        for metric, interval in metrics.items():
            if not isinstance(interval, dict) or "estimate" not in interval:
                continue
            rows.append({"section": section, "mode": mode, "metric": metric, **interval})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-run", type=Path, required=True)
    parser.add_argument("--generation-run", type=Path, required=True)
    parser.add_argument("--image-run", type=Path, required=True)
    parser.add_argument("--metadata-run", type=Path)
    parser.add_argument("--reranker-pool-run", type=Path)
    parser.add_argument("--chunk-run", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--permutation-resamples", type=int, default=50_000)
    args = parser.parse_args()

    retrieval_path = args.retrieval_run / "details.json"
    generation_answers = args.generation_run / "answers.json"
    generation_judgments = args.generation_run / "judgments.json"
    image_answers = args.image_run / "answers.json"
    image_judgments = args.image_run / "judgments.json"
    image_retrieval = args.image_run / "retrieval-results.json"
    inputs = [
        retrieval_path,
        generation_answers,
        generation_judgments,
        image_answers,
        image_judgments,
        image_retrieval,
    ]
    if args.metadata_run:
        inputs.append(args.metadata_run / "details.json")
    if args.reranker_pool_run:
        inputs.append(args.reranker_pool_run / "details.json")
    if args.chunk_run:
        inputs.append(args.chunk_run / "details.json")
    missing = [path for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing evaluation inputs: {missing}")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = create_run_dir(output_root)

    retrieval, retrieval_errors = retrieval_analysis(
        retrieval_path,
        bootstrap_resamples=args.bootstrap_resamples,
        permutation_resamples=args.permutation_resamples,
    )
    generation, generation_errors = generation_analysis(
        generation_answers,
        generation_judgments,
        bootstrap_resamples=args.bootstrap_resamples,
        permutation_resamples=args.permutation_resamples,
    )
    image, image_errors = image_analysis(
        image_answers,
        image_judgments,
        image_retrieval,
        bootstrap_resamples=args.bootstrap_resamples,
        permutation_resamples=args.permutation_resamples,
    )
    sensitivity = {}
    if args.metadata_run:
        sensitivity["metadata"] = rank_sensitivity_analysis(
            args.metadata_run / "details.json",
            kind="metadata",
            bootstrap_resamples=args.bootstrap_resamples,
            permutation_resamples=args.permutation_resamples,
        )
    if args.reranker_pool_run:
        sensitivity["reranker_pool"] = rank_sensitivity_analysis(
            args.reranker_pool_run / "details.json",
            kind="reranker_pool",
            bootstrap_resamples=args.bootstrap_resamples,
            permutation_resamples=args.permutation_resamples,
        )
    if args.chunk_run:
        sensitivity["chunk"] = rank_sensitivity_analysis(
            args.chunk_run / "details.json",
            kind="chunk",
            bootstrap_resamples=args.bootstrap_resamples,
            permutation_resamples=args.permutation_resamples,
        )

    all_errors = retrieval_errors + generation_errors + image_errors
    failure_only = [
        row
        for row in all_errors
        if row["category"]
        not in {
            "shared_success",
            "fusion_rescue",
            "reranker_promotion",
            "lexical_strength",
            "semantic_strength",
        }
    ]
    report = {
        "task": "statistical_and_failure_analysis",
        "git_sha": git_sha(),
        "seed": SEED,
        "bootstrap_resamples": args.bootstrap_resamples,
        "permutation_resamples": args.permutation_resamples,
        "retrieval": retrieval,
        "generation": generation,
        "image_qa": image,
        "sensitivity": sensitivity,
        "human_review_boundary": (
            "The labels received AI-assisted primary-source review but not independent clinical "
            "expert validation. Scientific claim review and the 30-document parser sample remain "
            "explicitly assigned to a person."
        ),
    }
    write_json(run_dir / "analysis.json", report)
    write_json(run_dir / "retrieval-analysis.json", retrieval)
    write_json(run_dir / "generation-analysis.json", generation)
    write_json(run_dir / "image-analysis.json", image)
    write_json(run_dir / "sensitivity-analysis.json", sensitivity)
    write_json(run_dir / "error-analysis.json", failure_only)
    write_csv(run_dir / "error-analysis.csv", failure_only)
    write_csv(
        run_dir / "confidence-intervals.csv",
        flatten_intervals("retrieval", retrieval["confidence_intervals"])
        + flatten_intervals("generation", generation["summaries"])
        + flatten_intervals("image_qa", image["summaries"])
        + [
            row
            for section_name, section in sensitivity.items()
            for row in flatten_intervals(
                section_name,
                section["confidence_intervals"],
            )
        ],
    )
    write_csv(
        run_dir / "paired-tests.csv",
        retrieval["paired_randomisation_tests"]
        + generation["paired_randomisation_tests"]
        + image["paired_randomisation_tests"]
        + [
            row
            for section in sensitivity.values()
            for row in section["paired_randomisation_tests"]
        ],
    )
    write_csv(run_dir / "judge-agreement.csv", generation["judge_repeat_agreement"])

    package_names = [
        "fastapi",
        "langchain",
        "langchain-google-genai",
        "numpy",
        "scipy",
        "scikit-learn",
        "sentence-transformers",
        "PyMuPDF",
        "Pillow",
        "httpx",
    ]
    packages = {}
    for name in package_names:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
        "inputs": [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in inputs
        ],
        "outputs": sorted(path.name for path in run_dir.iterdir()),
        "limitations": [
            "Confidence intervals quantify sampling uncertainty over the current cases only.",
            "Paired tests do not correct silver-label error or LLM-judge family bias.",
            "Automated error categories are screening labels, not expert scientific judgements.",
        ],
    }
    write_json(run_dir / "analysis-manifest.json", manifest)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
