#!/usr/bin/env python3
"""Create an AI-assisted claim-citation audit without impersonating human review."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from scripts.build_manual_review_pack import citation_claim_rows, write_csv
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from build_manual_review_pack import citation_claim_rows, write_csv


REVIEWER_TYPE = "AI-assisted primary-source semantic review (non-clinical)"
CHECKED_AT = "2026-08-10"


REVIEWS: list[dict[str, Any]] = [
    {
        "expected_keys": "lewis2020rag",
        "original_assessment": "supported_with_minor_scope_risk",
        "final_assessment": "supported",
        "evidence_note": (
            "The NeurIPS record supports the parametric/non-parametric architecture, "
            "retrieved Wikipedia passages and gains on knowledge-intensive tasks. The final "
            "wording avoids claiming that RAG itself guarantees inspectable proof."
        ),
        "primary_source_urls": (
            "https://papers.nips.cc/paper_files/paper/2020/hash/"
            "6b493230205f780e1bc26945df7481e5-Abstract.html"
        ),
    },
    {
        "expected_keys": "lewis2020rag",
        "original_assessment": "supported",
        "final_assessment": "supported",
        "evidence_note": (
            "Lewis et al. report more specific and factual generation than a parametric-only "
            "baseline; the limitation to Wikipedia experiments is stated as this project's "
            "scope judgement, not as a result of the cited paper."
        ),
        "primary_source_urls": (
            "https://papers.nips.cc/paper_files/paper/2020/hash/"
            "6b493230205f780e1bc26945df7481e5-Abstract.html"
        ),
    },
    {
        "expected_keys": (
            "li2023halueval; bouyamourn2023hallucinate; es2024ragas"
        ),
        "original_assessment": "partial",
        "final_assessment": "supported_after_revision",
        "evidence_note": (
            "HaluEval supports generation and recognition failures; Bouyamourn supports the "
            "evidential-closure argument. RAGAS was added for the requirement that retrieved "
            "context be relevant and used faithfully."
        ),
        "primary_source_urls": "; ".join(
            [
                "https://aclanthology.org/2023.emnlp-main.397/",
                "https://aclanthology.org/2023.emnlp-main.192/",
                "https://aclanthology.org/2024.eacl-demo.16/",
            ]
        ),
    },
    {
        "expected_keys": "robertson2009bm25; karpukhin2020dpr; reimers2019sbert",
        "original_assessment": "partial",
        "final_assessment": "supported_after_revision",
        "evidence_note": (
            "The final text limits BM25 to a lexical term-weighting signal, retains DPR's "
            "reported open-domain comparisons, and describes SBERT's independently encoded "
            "similarity search. Corpus-specific superiority is no longer asserted."
        ),
        "primary_source_urls": "; ".join(
            [
                "https://www.nowpublishers.com/article/DownloadEBook/INR-019",
                "https://aclanthology.org/2020.emnlp-main.550/",
                "https://aclanthology.org/D19-1410/",
            ]
        ),
    },
    {
        "expected_keys": "cormack2009rrf",
        "original_assessment": "supported_with_uncited_method_rationale",
        "final_assessment": "supported_after_revision",
        "evidence_note": (
            "The Waterloo-hosted paper defines RRF over reciprocal ranks. The comparison with "
            "weighted fusion is now explicitly described as this project's calibration rationale."
        ),
        "primary_source_urls": "https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf",
    },
    {
        "expected_keys": "shen2022vila; chen2023layout; blecher2023nougat",
        "original_assessment": "partial",
        "final_assessment": "supported_after_revision",
        "evidence_note": (
            "VILA supports line/block layout grouping, Chen et al. report degradation under "
            "layout distribution shifts, and Nougat is a visual OCR-to-markup model. Unsupported "
            "claims about discipline shifts and unspecified compute cost were removed."
        ),
        "primary_source_urls": "; ".join(
            [
                "https://aclanthology.org/2022.tacl-1.22/",
                "https://aclanthology.org/2023.findings-acl.844/",
                "https://proceedings.iclr.cc/paper_files/paper/2024/file/"
                "a39a9aceda771cded859ae7560530e09-Paper-Conference.pdf",
            ]
        ),
    },
    {
        "expected_keys": "liu2021dense; liu2024lost",
        "original_assessment": "partial",
        "final_assessment": "supported_after_revision",
        "evidence_note": (
            "Dense Hierarchical Retrieval directly identifies local and partial context caused "
            "by passage splitting; Lost in the Middle reports poorer access to evidence placed "
            "mid-context. Generic chunk-quality claims were narrowed to these findings."
        ),
        "primary_source_urls": "; ".join(
            [
                "https://aclanthology.org/2021.findings-emnlp.19/",
                "https://aclanthology.org/2024.tacl-1.9/",
            ]
        ),
    },
    {
        "expected_keys": "singh2023scirepeval",
        "original_assessment": "partial",
        "final_assessment": "supported_after_revision",
        "evidence_note": (
            "SciRepEval supports cross-format generalisation difficulty for scientific document "
            "representations. Titles, sections and provenance are now presented as system design "
            "choices rather than conclusions attributed to SciRepEval."
        ),
        "primary_source_urls": "https://aclanthology.org/2023.emnlp-main.338/",
    },
    {
        "expected_keys": "es2024ragas; saadfalcon2024ares; amugongo2025healthcare",
        "original_assessment": "supported_with_broad_wording",
        "final_assessment": "supported_after_revision",
        "evidence_note": (
            "RAGAS separates retrieval/context, faithful use and generation quality; ARES uses "
            "synthetic judge training plus a small human-labelled set. The healthcare review "
            "specifically reports a lack of standardised evaluation frameworks."
        ),
        "primary_source_urls": "; ".join(
            [
                "https://aclanthology.org/2024.eacl-demo.16/",
                "https://aclanthology.org/2024.naacl-long.20/",
                "https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000877",
            ]
        ),
    },
    {
        "expected_keys": "radford2021clip; li2022blip; wu2024scimmir",
        "original_assessment": "supported_with_broad_wording",
        "final_assessment": "supported_after_revision",
        "evidence_note": (
            "CLIP supports large-scale transferable image-text learning; BLIP supports filtered "
            "caption bootstrapping for understanding and generation. SciMMIR directly contrasts "
            "scientific captions with generic activities and scenery."
        ),
        "primary_source_urls": "; ".join(
            [
                "https://proceedings.mlr.press/v139/radford21a.html",
                "https://proceedings.mlr.press/v162/li22n.html",
                "https://aclanthology.org/2024.findings-acl.746/",
            ]
        ),
    },
    {
        "expected_keys": (
            "mattay2024mr; lena2021interleaved; rohfritsch2024ultrasound; "
            "ring2012infrared; lahiri2012medical; stanley2024acute"
        ),
        "original_assessment": "partial",
        "final_assessment": "supported_after_revision",
        "evidence_note": (
            "The final wording distinguishes real-time MR monitoring, the experimental and "
            "tissue-dependent ultrasound backscatter result, and non-contact surface infrared "
            "mapping with protocol/environment constraints. Cost and universal modality claims "
            "were removed. This is literature support, not clinical endorsement."
        ),
        "primary_source_urls": "; ".join(
            [
                "https://pubmed.ncbi.nlm.nih.gov/38123912/",
                "https://pubmed.ncbi.nlm.nih.gov/34061390/",
                "https://pubmed.ncbi.nlm.nih.gov/38850600/",
                "https://pubmed.ncbi.nlm.nih.gov/22370242/",
                "https://pubmed.ncbi.nlm.nih.gov/32288544/",
                "https://pubmed.ncbi.nlm.nih.gov/38983367/",
            ]
        ),
    },
    {
        "expected_keys": (
            "robertson2009bm25; karpukhin2020dpr; reimers2019sbert; "
            "cormack2009rrf"
        ),
        "original_assessment": "supported",
        "final_assessment": "supported",
        "evidence_note": (
            "The cited papers distinguish lexical term weighting, independently encoded "
            "semantic representations and reciprocal-rank fusion. Numerical improvements and "
            "their corpus-specific boundary come from the saved project experiment."
        ),
        "primary_source_urls": "; ".join(
            [
                "https://www.nowpublishers.com/article/DownloadEBook/INR-019",
                "https://aclanthology.org/2020.emnlp-main.550/",
                "https://aclanthology.org/D19-1410/",
                "https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf",
            ]
        ),
    },
    {
        "expected_keys": "liu2021dense; liu2024lost",
        "original_assessment": "partial",
        "final_assessment": "supported_after_revision",
        "evidence_note": (
            "The paragraph now links the project's measured storage/accuracy trade-off to two "
            "supported risks: lost document context after splitting and unreliable use of "
            "evidence in long contexts. It no longer attributes improved focus to either paper."
        ),
        "primary_source_urls": "; ".join(
            [
                "https://aclanthology.org/2021.findings-emnlp.19/",
                "https://aclanthology.org/2024.tacl-1.9/",
            ]
        ),
    },
    {
        "expected_keys": "es2024ragas; saadfalcon2024ares",
        "original_assessment": "supported",
        "final_assessment": "supported",
        "evidence_note": (
            "Both frameworks explicitly evaluate separate RAG components or dimensions. The "
            "reported percentage and significance results are traced to the saved project run, "
            "not attributed to the cited frameworks."
        ),
        "primary_source_urls": "; ".join(
            [
                "https://aclanthology.org/2024.eacl-demo.16/",
                "https://aclanthology.org/2024.naacl-long.20/",
            ]
        ),
    },
    {
        "expected_keys": (
            "li2023halueval; bouyamourn2023hallucinate; amugongo2025healthcare"
        ),
        "original_assessment": "partial",
        "final_assessment": "supported_after_revision",
        "evidence_note": (
            "HaluEval and evidential closure support the general hallucination mechanism; the "
            "healthcare review was added because it explicitly reports incorrect or unsupported "
            "answers even with RAG. The paragraph does not claim clinical validation."
        ),
        "primary_source_urls": "; ".join(
            [
                "https://aclanthology.org/2023.emnlp-main.397/",
                "https://aclanthology.org/2023.emnlp-main.192/",
                "https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000877",
            ]
        ),
    },
    {
        "expected_keys": "radford2021clip; li2022blip; wu2024scimmir",
        "original_assessment": "supported_with_inference",
        "final_assessment": "supported_after_revision",
        "evidence_note": (
            "The final text presents the ceiling effect as consistent with, rather than proof "
            "of, SciMMIR's documented scientific-domain gap. CLIP and BLIP supply the generic "
            "image-language comparison; latency and significance are project results."
        ),
        "primary_source_urls": "; ".join(
            [
                "https://proceedings.mlr.press/v139/radford21a.html",
                "https://proceedings.mlr.press/v162/li22n.html",
                "https://aclanthology.org/2024.findings-acl.746/",
            ]
        ),
    },
    {
        "expected_keys": "shen2022vila; chen2023layout",
        "original_assessment": "supported",
        "final_assessment": "supported",
        "evidence_note": (
            "VILA demonstrates the importance of visual layout groups and Chen et al. quantify "
            "degradation under layout distribution shift. The observed parser defects and "
            "sample-size limitation are explicitly reported as project audit results."
        ),
        "primary_source_urls": "; ".join(
            [
                "https://aclanthology.org/2022.tacl-1.22/",
                "https://aclanthology.org/2023.findings-acl.844/",
            ]
        ),
    },
]


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    revised = sum(row["final_assessment"] == "supported_after_revision" for row in rows)
    lines = [
        "# AI-Assisted Claim-Citation Review",
        "",
        "## Boundary",
        "",
        "This review checks whether the cited primary publications support the report's",
        "wording. It is an AI-assisted technical literature review, not independent human,",
        "medical, legal or clinical sign-off. No private source text is reproduced here.",
        "",
        "## Result",
        "",
        f"- Claim groups checked: {len(rows)}",
        f"- Supported after final wording review: {len(rows)}",
        f"- Groups narrowed or re-cited: {revised}",
        "- Unsupported groups remaining: 0",
        "",
        "The revisions removed or narrowed claims about universal dense-retrieval",
        "superiority, discipline-level PDF shifts, unspecified neural-parser cost, generic",
        "chunk effects, unresolved healthcare evaluation, and modality-wide clinical claims.",
        "The independent human review form remains blank by design.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--human-template", type=Path, required=True)
    args = parser.parse_args()

    claim_rows = citation_claim_rows(args.tex)
    if len(claim_rows) != len(REVIEWS):
        raise ValueError(
            f"Expected {len(REVIEWS)} cited claim groups, found {len(claim_rows)}"
        )

    audited_rows: list[dict[str, Any]] = []
    for index, (claim, review) in enumerate(
        zip(claim_rows, REVIEWS, strict=True),
        start=1,
    ):
        if claim["citation_keys"] != review["expected_keys"]:
            raise ValueError(
                f"Claim {index} citation mismatch: {claim['citation_keys']!r} != "
                f"{review['expected_keys']!r}"
            )
        audited_rows.append(
            {
                "claim_id": f"c{index:02d}",
                "section": claim["section"],
                "source_line": claim["source_line"],
                "citation_keys": claim["citation_keys"],
                "final_claim_text": claim["claim_text"],
                "original_assessment": review["original_assessment"],
                "final_assessment": review["final_assessment"],
                "scope_accurate": "yes",
                "evidence_note": review["evidence_note"],
                "primary_source_urls": review["primary_source_urls"],
                "reviewer_type": REVIEWER_TYPE,
                "checked_at": CHECKED_AT,
                "independent_human_review": "not_performed",
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "04-claim-citation-ai-review.csv", audited_rows)
    write_summary(args.output / "04-claim-citation-ai-review-summary.md", audited_rows)

    # Refresh line numbers and final wording while preserving blank human judgements.
    write_csv(args.human_template, claim_rows)
    print(
        f"Reviewed {len(audited_rows)} claim groups; refreshed independent template at "
        f"{args.human_template}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
