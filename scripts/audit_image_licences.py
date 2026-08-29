#!/usr/bin/env python3
"""Verify selected image-QA figures against article-level Creative Commons metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx


THIRD_PARTY_RE = re.compile(
    r"\b(adapted|reproduced|reprinted|modified)\s+from\b|"
    r"\bpermission\s+(?:from|of)\b|©",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    candidates = {
        str(row["candidate_id"]): row
        for row in json.loads(args.candidates.read_text(encoding="utf-8"))
    }
    selected = []
    for item in selection:
        candidate_id = str(item["candidate_id"])
        if candidate_id not in candidates:
            raise KeyError(f"Unknown selected candidate: {candidate_id}")
        selected.append({**candidates[candidate_id], **item})

    headers = {
        "User-Agent": "UoB-MSc-RAG-image-licence-audit/1.0 "
        "(mailto:student@bham.ac.uk)"
    }
    article_metadata: dict[str, dict[str, Any]] = {}
    with httpx.Client(timeout=30, follow_redirects=True, headers=headers) as client:
        for doi in sorted({str(row.get("doi") or "").lower() for row in selected}):
            response = client.get(
                f"https://api.crossref.org/works/{quote(doi, safe='')}"
            )
            if response.status_code != 200:
                article_metadata[doi] = {
                    "registry_verified": False,
                    "status_code": response.status_code,
                }
                continue
            item = response.json()["message"]
            licences = [
                str(licence.get("URL") or "")
                for licence in item.get("license") or []
            ]
            article_metadata[doi] = {
                "registry_verified": True,
                "title": (item.get("title") or [""])[0],
                "publisher": item.get("publisher"),
                "resource_url": item.get("URL"),
                "licence_urls": licences,
                "cc_by_4_verified": any(
                    "creativecommons.org/licenses/by/4.0" in url.lower()
                    for url in licences
                ),
            }

    rows = []
    for item in selected:
        doi = str(item.get("doi") or "").lower()
        metadata = article_metadata[doi]
        image_path = Path(item["image_path"])
        caption = str(item.get("caption") or "")
        third_party_signal = bool(THIRD_PARTY_RE.search(caption))
        rows.append(
            {
                "candidate_id": item["candidate_id"],
                "category": item.get("category"),
                "title": item.get("title"),
                "doi": doi,
                "page": item.get("page"),
                "figure_number": item.get("figure_number"),
                "image_sha256": sha256(image_path),
                "article_licence": "CC BY 4.0" if metadata.get("cc_by_4_verified") else None,
                "licence_url": (
                    next(
                        (
                            url
                            for url in metadata.get("licence_urls") or []
                            if "creativecommons.org/licenses/by/4.0" in url.lower()
                        ),
                        None,
                    )
                ),
                "registry_verified": metadata.get("registry_verified"),
                "cc_by_4_verified": metadata.get("cc_by_4_verified"),
                "third_party_caption_signal": third_party_signal,
                "third_party_material_review": (
                    "manual_check_required"
                    if third_party_signal
                    else "no_automated_caption_signal"
                ),
                "attribution": (
                    f"{item.get('figure_number')}, {item.get('title')} "
                    f"({item.get('year')}), DOI {doi}, CC BY 4.0."
                ),
            }
        )

    report = {
        "summary": {
            "selected_figure_count": len(rows),
            "unique_article_count": len(article_metadata),
            "cc_by_4_verified_count": sum(
                bool(row["cc_by_4_verified"]) for row in rows
            ),
            "third_party_caption_signal_count": sum(
                bool(row["third_party_caption_signal"]) for row in rows
            ),
            "scope_note": (
                "Crossref confirms the version-of-record article licence. Automated caption "
                "screening cannot exclude an unlabelled third-party element inside a figure."
            ),
        },
        "articles": article_metadata,
        "figures": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output.with_suffix(".csv"), rows)
    summary = report["summary"]
    print(
        f"Verified CC BY 4.0 for {summary['cc_by_4_verified_count']}/"
        f"{summary['selected_figure_count']} figures from "
        f"{summary['unique_article_count']} articles."
    )
    return 0 if (
        summary["cc_by_4_verified_count"] == summary["selected_figure_count"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
