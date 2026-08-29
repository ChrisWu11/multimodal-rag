#!/usr/bin/env python3
"""Package privacy-safe final-report evaluation artefacts for review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "results" / "final-report"


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


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sanitise_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: sanitise_json(item)
            for key, item in value.items()
            if key not in {"pdf_path", "image_path"}
        }
    if isinstance(value, list):
        return [sanitise_json(item) for item in value]
    if isinstance(value, str) and (
        value.startswith("/Users/")
        or value.startswith(str(PROJECT_ROOT))
    ):
        return "<redacted-local-path>"
    return value


PDF_AUDIT_SAFE_COLUMNS = [
    "doc_id",
    "year",
    "page_count",
    "extracted_pages",
    "actual_page_count",
    "reading_order_correct",
    "pages_complete",
    "headings_and_sections_correct",
    "header_footer_noise_acceptable",
    "formula_table_corruption_acceptable",
    "metadata_correct",
    "decision",
    "reviewer_notes",
    "reference_stop_page",
    "post_reference_section_pages",
    "multi_section_pages",
    "unknown_section_pages",
    "table_pages",
    "weak_table_pages",
]


def copy_sanitised_pdf_audit(source: Path, destination: Path) -> None:
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PDF_AUDIT_SAFE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in PDF_AUDIT_SAFE_COLUMNS})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-run", type=Path, required=True)
    parser.add_argument("--retrieval-run", type=Path, required=True)
    parser.add_argument("--chunk-run", type=Path, required=True)
    parser.add_argument("--metadata-run", type=Path, required=True)
    parser.add_argument("--reranker-pool-run", type=Path, required=True)
    parser.add_argument("--generation-run", type=Path, required=True)
    parser.add_argument("--image-run", type=Path, required=True)
    parser.add_argument("--analysis-run", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--image-licence-audit", type=Path, required=True)
    parser.add_argument("--pdf-audit-run", type=Path, required=True)
    parser.add_argument(
        "--image-rights-precheck",
        type=Path,
        default=(
            PROJECT_ROOT
            / "evaluation"
            / "manual-review"
            / "03-image-rights-ai-precheck.csv"
        ),
    )
    parser.add_argument(
        "--image-rights-disposition",
        type=Path,
        default=(
            PROJECT_ROOT
            / "evaluation"
            / "manual-review"
            / "03-image-rights-ai-disposition.csv"
        ),
    )
    parser.add_argument(
        "--claim-citation-review",
        type=Path,
        default=(
            PROJECT_ROOT
            / "evaluation"
            / "manual-review"
            / "04-claim-citation-ai-review.csv"
        ),
    )
    parser.add_argument("--result-claims-audit", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    selections = [
        (args.corpus_run / "summary.json", "corpus-summary.json"),
        (args.retrieval_run / "summary.json", "retrieval-summary.json"),
        (args.retrieval_run / "details.json", "retrieval-details.json"),
        (args.chunk_run / "summary.json", "chunk-ablation-summary.json"),
        (args.chunk_run / "details.json", "chunk-ablation-details.json"),
        (args.metadata_run / "summary.json", "metadata-ablation-summary.json"),
        (args.metadata_run / "details.json", "metadata-ablation-details.json"),
        (
            args.reranker_pool_run / "summary.json",
            "reranker-pool-ablation-summary.json",
        ),
        (
            args.reranker_pool_run / "details.json",
            "reranker-pool-ablation-details.json",
        ),
        (args.generation_run / "summary.json", "generation-summary.json"),
        (args.image_run / "summary.json", "image-qa-summary.json"),
        (args.analysis_run / "analysis.json", "statistical-analysis.json"),
        (args.analysis_run / "confidence-intervals.csv", "confidence-intervals.csv"),
        (args.analysis_run / "paired-tests.csv", "paired-tests.csv"),
        (args.analysis_run / "judge-agreement.csv", "judge-agreement.csv"),
        (args.analysis_run / "error-analysis.csv", "error-analysis.csv"),
        (
            args.analysis_run / "sensitivity-analysis.json",
            "sensitivity-analysis.json",
        ),
        (args.reference_audit, "reference-audit.json"),
        (args.image_licence_audit, "image-licence-audit.json"),
        (args.image_rights_precheck, "image-rights-ai-precheck.csv"),
        (args.image_rights_disposition, "image-rights-ai-disposition.csv"),
        (args.claim_citation_review, "claim-citation-ai-review.csv"),
        (
            args.pdf_audit_run / "02-pdf-technical-ai-review-summary.json",
            "pdf-technical-audit-summary.json",
        ),
        (
            args.pdf_audit_run / "02-pdf-technical-ai-review.csv",
            "pdf-technical-ai-review.csv",
        ),
    ]
    if args.result_claims_audit is not None:
        selections.append((args.result_claims_audit, "report-result-claims-audit.json"))
    missing = [path for path, _ in selections if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing package inputs: {missing}")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_files = []
    for source, name in selections:
        destination = output / name
        if name == "pdf-technical-ai-review.csv":
            copy_sanitised_pdf_audit(source, destination)
        elif source.suffix.lower() == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
            write_json(destination, sanitise_json(payload))
        else:
            shutil.copy2(source, destination)
        manifest_files.append(
            {
                "file": name,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    write_json(
        output / "manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluated_git_sha": git_sha(),
            "files": manifest_files,
            "privacy_boundary": {
                "included": (
                    "Aggregate metrics, per-question ranks and identifiers, statistical "
                    "outputs, bibliographic verification, article-level licence records "
                    "and author-led claim-citation, source-page rights, publication "
                    "disposition and PDF parsing audits with subsequent Codex cross-checking."
                ),
                "excluded": (
                    "Private PDFs, extracted evidence text, source figures, embeddings, "
                    "databases, local paths, generated answers, raw judge payloads and API keys."
                ),
            },
        },
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
