#!/usr/bin/env python3
"""Verify that headline LaTeX claims match the saved evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mode(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(row for row in rows if row.get("mode") == name)


def check_fragment(
    checks: list[dict[str, Any]],
    tex: str,
    label: str,
    fragment: str,
) -> None:
    present = fragment in tex
    checks.append({"label": label, "expected_fragment": fragment, "present": present})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--silver-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tex = re.sub(r"\s+", " ", args.tex.read_text(encoding="utf-8"))
    corpus = load_json(args.results / "corpus-summary.json")
    retrieval = load_json(args.results / "retrieval-summary.json")
    chunks = load_json(args.results / "chunk-ablation-summary.json")
    analysis = load_json(args.results / "statistical-analysis.json")
    image_summary = load_json(args.results / "image-qa-summary.json")
    pdf_audit = load_json(args.results / "pdf-technical-audit-summary.json")

    lexical = mode(retrieval["summaries"], "fts_bm25")
    dense = mode(retrieval["summaries"], "st_dense")
    rrf = mode(retrieval["summaries"], "st_hybrid_rrf")
    lexical_rerank = mode(retrieval["summaries"], "rerank_fts_top30")
    c350 = next(row for row in chunks["summaries"] if row["configuration"] == "w350_o50")
    c650 = next(row for row in chunks["summaries"] if row["configuration"] == "w650_o90")
    generation = analysis["generation"]["summaries"]
    images = analysis["image_qa"]["summaries"]
    raw_images = {row["mode"]: row for row in image_summary["summaries"]}

    with args.silver_review.open(encoding="utf-8", newline="") as handle:
        silver_rows = list(csv.DictReader(handle))
    review_counts: dict[str, int] = {}
    for row in silver_rows:
        decision = row["decision"]
        review_counts[decision] = review_counts.get(decision, 0) + 1

    checks: list[dict[str, Any]] = []
    check_fragment(
        checks,
        tex,
        "corpus counts",
        (
            f"{corpus['document_count']} papers, {corpus['page_record_count']:,} page records "
            f"and {corpus['chunk_count']:,} chunks"
        ),
    )
    check_fragment(
        checks,
        tex,
        "corpus metadata rates",
        (
            f"{corpus['metadata_coverage']['doi'] * 100:.2f}\\% of chunks and "
            f"{corpus['unknown_section_chunk_rate'] * 100:.2f}\\% had an unknown section"
        ),
    )
    check_fragment(
        checks,
        tex,
        "silver-label decisions",
        (
            f"{review_counts['approved']} were approved and "
            f"{review_counts['corrected']} corrected"
        ),
    )
    check_fragment(
        checks,
        tex,
        "lexical retrieval",
        f"Lexical MRR and Recall@5 were {lexical['mrr']:.3f} and {lexical['recall_at_5']:.2f}",
    )
    check_fragment(
        checks,
        tex,
        "dense retrieval",
        f"dense retrieval reached {dense['mrr']:.3f} and {dense['recall_at_5']:.2f}",
    )
    check_fragment(
        checks,
        tex,
        "RRF retrieval",
        f"MRR {rrf['mrr']:.3f}",
    )
    check_fragment(
        checks,
        tex,
        "lexical reranking",
        f"Lexical reranking raised MRR to {lexical_rerank['mrr']:.3f}",
    )
    check_fragment(
        checks,
        tex,
        "chunk ablation",
        (
            f"The 350/50 condition created {c350['chunk_count']:,} chunks and achieved the "
            f"best MRR ({c350['mrr']:.3f}), Recall@5 ({c350['recall_at_5']:.2f})"
        ),
    )
    check_fragment(
        checks,
        tex,
        "production chunks",
        (
            f"Production 650/90 created {c650['chunk_count']:,} chunks and reached "
            f"{c650['mrr']:.3f}, {c650['recall_at_5']:.2f}"
        ),
    )

    generation_labels = {
        "llm_only": "LLM only",
        "dense_rag": "Dense RAG",
        "hybrid_reranked_rag": "Hybrid reranked RAG",
    }
    for mode_name, display in generation_labels.items():
        row = generation[mode_name]
        fragment = (
            f"{display} & {row['correctness']['estimate']:.3f} & "
            f"{row['faithfulness']['estimate']:.3f} & "
            f"{row['completeness']['estimate']:.3f} & "
            f"{row['unsupported_claim_rate']['estimate']:.3f} & "
            f"{row['abstention_accuracy']['estimate']:.3f}"
        )
        check_fragment(checks, tex, f"generation row: {mode_name}", fragment)

    image_labels = {
        "text_only": "Text only",
        "basic_metadata": "Basic metadata",
        "vision_summary": "Vision summary",
    }
    for mode_name, display in image_labels.items():
        row = images[mode_name]
        fragment = (
            f"{display} & {row['source_recall_at_5']['estimate']:.3f} & "
            f"{row['caption_coverage']['estimate']:.3f} & "
            f"{row['citation_quality']['estimate']:.3f} & "
            f"{row['response_structure']['estimate']:.3f} & "
            f"{row['visual_contradiction_count']}"
        )
        check_fragment(checks, tex, f"image row: {mode_name}", fragment)

    check_fragment(
        checks,
        tex,
        "visual inference counts",
        (
            "three unsupported visual inferences for text-only answers, one with basic "
            "metadata and none with the vision summary"
        ),
    )
    assert raw_images["text_only"]["unsupported_visual_inferences"] == 3
    assert raw_images["basic_metadata"]["unsupported_visual_inferences"] == 1
    assert raw_images["vision_summary"]["unsupported_visual_inferences"] == 0

    check_fragment(
        checks,
        tex,
        "PDF page audit",
        (
            f"all {pdf_audit['pdf_page_count']} physical pages from "
            f"{pdf_audit['document_count']} PDFs"
        ),
    )
    decisions = pdf_audit["decision_counts"]
    check_fragment(
        checks,
        tex,
        "PDF decisions",
        (
            f"{decisions['approved']} approved, "
            f"{decisions['approved_with_minor_issues']} minor, and "
            f"{decisions['needs_rebuild']} rebuild"
        ),
    )

    failed = [check for check in checks if not check["present"]]
    payload = {
        "status": "passed" if not failed else "failed",
        "checks": checks,
        "passed": len(checks) - len(failed),
        "failed": len(failed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
