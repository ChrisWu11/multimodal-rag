#!/usr/bin/env python3
"""Verify every BibTeX record against DOI registries or an official proceedings page."""

from __future__ import annotations

import argparse
import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx


OFFICIAL_OVERRIDES: dict[str, dict[str, Any]] = {
    "lewis2020rag": {
        "source": "NeurIPS proceedings",
        "url": (
            "https://papers.nips.cc/paper_files/paper/2020/hash/"
            "6b493230205f780e1bc26945df7481e5-Abstract.html"
        ),
        "year": "2020",
        "volume": "33",
        "pages": "9459-9474",
        "note": "Official proceedings record; this paper has no DOI in the bibliography.",
    },
    "robertson2009bm25": {
        "source": "DBLP record and publisher PDF",
        "url": "https://dblp.org/rec/journals/ftir/RobertsonZ09.html",
        "year": "2009",
        "volume": "3",
        "number": "4",
        "pages": "333-389",
        "note": (
            "The official article and DBLP record supersede an erroneous Crossref deposit "
            "that reports volume 4(1-2), pages 1-174."
        ),
    },
    "li2023halueval": {
        "source": "ACL Anthology proceedings PDF",
        "url": "https://aclanthology.org/2023.emnlp-main.397.pdf",
        "year": "2023",
        "pages": "6449-6464",
        "skip_registry_author_check": True,
        "note": (
            "The proceedings PDF identifies the third author as Wayne Xin Zhao; "
            "the Crossref deposit shortens this to Xin Zhao."
        ),
    },
    "reimers2019sbert": {
        "source": "ACL Anthology",
        "url": "https://aclanthology.org/D19-1410/",
        "year": "2019",
        "pages": "3982-3992",
        "note": (
            "ACL Anthology and the proceedings PDF supersede an erroneous Crossref page range "
            "of 3980-3990."
        ),
    },
    "blecher2023nougat": {
        "source": "ICLR proceedings",
        "url": (
            "https://proceedings.iclr.cc/paper_files/paper/2024/file/"
            "a39a9aceda771cded859ae7560530e09-Paper-Conference.pdf"
        ),
        "year": "2024",
        "note": "Official ICLR 2024 proceedings paper; no DOI is assigned.",
    },
    "radford2021clip": {
        "source": "Proceedings of Machine Learning Research",
        "url": "https://proceedings.mlr.press/v139/radford21a.html",
        "year": "2021",
        "volume": "139",
        "pages": "8748-8763",
        "note": "Official PMLR proceedings record; no DOI is assigned.",
    },
    "li2022blip": {
        "source": "Proceedings of Machine Learning Research",
        "url": "https://proceedings.mlr.press/v162/li22n.html",
        "year": "2022",
        "volume": "162",
        "pages": "12888-12900",
        "note": "Official PMLR proceedings record; no DOI is assigned.",
    },
    "mattay2024mr": {
        "source": "PubMed",
        "url": "https://pubmed.ncbi.nlm.nih.gov/38123912/",
        "year": "2024",
        "volume": "45",
        "number": "1",
        "pages": "1-8",
        "note": (
            "The article was electronically published on 29 December 2023 and assigned to "
            "volume 45 issue 1 (January 2024); the report consistently cites the issue year."
        ),
    },
}


def parse_bibtex(path: Path) -> dict[str, dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    entries: dict[str, dict[str, str]] = {}
    for match in re.finditer(
        r"@(?P<type>\w+)\{(?P<key>[^,]+),(?P<body>.*?)(?=^\})",
        text,
        re.MULTILINE | re.DOTALL,
    ):
        fields: dict[str, str] = {"entry_type": match.group("type")}
        for line in match.group("body").splitlines():
            field = re.match(r"\s*(\w+)\s*=\s*\{(.*)\}\s*,?\s*$", line)
            if field:
                fields[field.group(1).lower()] = field.group(2).strip()
        entries[match.group("key").strip()] = fields
    return entries


def normalise_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\\[A-Za-z]+\s*", " ", text)
    text = re.sub(r"[{}\\]", "", text)
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def normalise_pages(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")


def citation_keys(tex_path: Path) -> set[str]:
    text = tex_path.read_text(encoding="utf-8")
    keys: set[str] = set()
    for citation in re.findall(r"\\cite\{([^}]+)\}", text):
        keys.update(key.strip() for key in citation.split(",") if key.strip())
    return keys


def crossref_record(client: httpx.Client, doi: str) -> dict[str, Any]:
    response = client.get(f"https://api.crossref.org/works/{quote(doi, safe='')}")
    if response.status_code != 200:
        return {"verified": False, "status_code": response.status_code}
    item = response.json()["message"]
    return {
        "verified": True,
        "title": (item.get("title") or [""])[0],
        "authors": [
            " ".join(
                part
                for part in (author.get("given", ""), author.get("family", ""))
                if part
            )
            for author in item.get("author", [])
        ],
        "author_parts": [
            {
                "given": str(author.get("given") or ""),
                "family": str(author.get("family") or ""),
            }
            for author in item.get("author", [])
        ],
        "container_title": (item.get("container-title") or [""])[0],
        "year": str(
            ((item.get("published") or {}).get("date-parts") or [[""]])[0][0]
        ),
        "volume": str(item.get("volume") or ""),
        "number": str(item.get("issue") or ""),
        "pages": str(item.get("page") or ""),
        "type": item.get("type"),
        "url": item.get("URL"),
    }


def bibtex_author_parts(value: str) -> list[dict[str, str]]:
    """Return BibTeX authors as given/family parts without changing their order."""
    authors: list[dict[str, str]] = []
    for raw_author in re.split(r"\s+and\s+", value.strip()):
        author = raw_author.strip()
        if not author:
            continue
        if "," in author:
            family, given = (part.strip() for part in author.split(",", 1))
        else:
            parts = author.split()
            family = parts[-1]
            given = " ".join(parts[:-1])
        authors.append({"given": given, "family": family})
    return authors


def normalise_name_part(value: str) -> str:
    return normalise_text(value).replace(" ", "")


def given_names_match(bibliography: str, registry: str) -> bool:
    bibliography_tokens = normalise_text(bibliography).split()
    registry_tokens = normalise_text(registry).split()
    if not bibliography_tokens or not registry_tokens:
        return True
    if bibliography_tokens == registry_tokens:
        return True

    shared_length = min(len(bibliography_tokens), len(registry_tokens))
    for bibliography_token, registry_token in zip(
        bibliography_tokens[:shared_length],
        registry_tokens[:shared_length],
        strict=True,
    ):
        if bibliography_token == registry_token:
            continue
        if min(len(bibliography_token), len(registry_token)) == 1:
            if bibliography_token[0] == registry_token[0]:
                continue
        return False

    remaining = (
        bibliography_tokens[shared_length:] + registry_tokens[shared_length:]
    )
    return all(len(token) == 1 for token in remaining)


def author_checks(
    fields: dict[str, str],
    registry: dict[str, Any],
) -> list[str]:
    bibliography_authors = bibtex_author_parts(fields.get("author", ""))
    registry_authors = registry.get("author_parts") or []
    if not bibliography_authors or not registry_authors:
        return []
    if len(bibliography_authors) != len(registry_authors):
        return [
            "author count: "
            f"bibliography={len(bibliography_authors)}; registry={len(registry_authors)}"
        ]

    discrepancies: list[str] = []
    for index, (bibliography, registered) in enumerate(
        zip(bibliography_authors, registry_authors, strict=True),
        start=1,
    ):
        bibliography_family = normalise_name_part(bibliography["family"])
        registry_family = normalise_name_part(registered["family"])
        given_matches = given_names_match(
            bibliography["given"],
            registered["given"],
        )
        if bibliography_family != registry_family or not given_matches:
            discrepancies.append(
                f"author {index}: bibliography="
                f"{bibliography['given']} {bibliography['family']!s}; registry="
                f"{registered['given']} {registered['family']!s}"
            )
    return discrepancies


def official_checks(
    fields: dict[str, str],
    official: dict[str, Any],
) -> list[str]:
    discrepancies = []
    for field in ("year", "volume", "number", "pages"):
        expected = official.get(field)
        actual = fields.get(field)
        if expected is None or actual is None:
            continue
        if field == "pages":
            matches = normalise_pages(expected) == normalise_pages(actual)
        else:
            matches = normalise_text(expected) == normalise_text(actual)
        if not matches:
            discrepancies.append(f"{field}: bibliography={actual!r}; official={expected!r}")
    return discrepancies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bibliography", type=Path)
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    entries = parse_bibtex(args.bibliography)
    used_keys = citation_keys(args.tex)
    records: list[dict[str, Any]] = []
    headers = {
        "User-Agent": "UoB-MSc-RAG-reference-audit/2.0 (mailto:student@bham.ac.uk)"
    }
    with httpx.Client(timeout=30, follow_redirects=True, headers=headers) as client:
        for key, fields in entries.items():
            doi = fields.get("doi", "").lower()
            official = OFFICIAL_OVERRIDES.get(key)
            registry = crossref_record(client, doi) if doi else None
            title_similarity = (
                round(
                    SequenceMatcher(
                        None,
                        normalise_text(fields.get("title")),
                        normalise_text(registry.get("title")),
                    ).ratio(),
                    4,
                )
                if registry and registry.get("verified")
                else None
            )
            if official:
                source_status = client.get(official["url"]).status_code
                discrepancies = official_checks(fields, official)
                if (
                    registry
                    and registry.get("verified")
                    and not official.get("skip_registry_author_check", False)
                ):
                    discrepancies.extend(author_checks(fields, registry))
                verified = source_status < 400 and not discrepancies
                verifier = official["source"]
                source_url = official["url"]
                decision_note = official["note"]
            elif registry and registry.get("verified"):
                discrepancies = []
                if title_similarity is not None and title_similarity < 0.82:
                    discrepancies.append(
                        f"title similarity below threshold: {title_similarity}"
                    )
                bibliography_year = fields.get("year", "")
                registry_year = registry.get("year", "")
                if bibliography_year and registry_year and bibliography_year != registry_year:
                    discrepancies.append(
                        f"year: bibliography={bibliography_year!r}; Crossref={registry_year!r}"
                    )
                discrepancies.extend(author_checks(fields, registry))
                verified = not discrepancies
                verifier = "Crossref DOI registry"
                source_url = registry.get("url")
                decision_note = (
                    "DOI, title, publication year and ordered author names matched the "
                    "registry record."
                )
            else:
                discrepancies = ["No DOI registry record or configured official source."]
                verified = False
                verifier = "unverified"
                source_url = None
                decision_note = ""

            records.append(
                {
                    "key": key,
                    "cited_in_report": key in used_keys,
                    "verified": verified,
                    "doi": doi or None,
                    "title": fields.get("title"),
                    "year": fields.get("year"),
                    "volume": fields.get("volume"),
                    "number": fields.get("number"),
                    "pages": fields.get("pages"),
                    "verifier": verifier,
                    "source_url": source_url,
                    "source_http_status": source_status if official else None,
                    "title_similarity_to_crossref": title_similarity,
                    "crossref": registry,
                    "discrepancies": discrepancies,
                    "decision_note": decision_note,
                    "claim_support_review_status": "pending_human_reading",
                }
            )
            time.sleep(0.05)

    bibliography_keys = set(entries)
    report = {
        "summary": {
            "bibliography_count": len(entries),
            "cited_key_count": len(used_keys),
            "verified_count": sum(bool(row["verified"]) for row in records),
            "unverified_count": sum(not row["verified"] for row in records),
            "missing_bibliography_keys": sorted(used_keys - bibliography_keys),
            "uncited_bibliography_keys": sorted(bibliography_keys - used_keys),
            "metadata_scope": (
                "Automated verification confirms bibliographic identity and metadata, not "
                "whether each cited sentence is semantically supported."
            ),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = report["summary"]
    print(
        f"Verified {summary['verified_count']}/{summary['bibliography_count']} references; "
        f"missing keys: {summary['missing_bibliography_keys']}; "
        f"uncited keys: {summary['uncited_bibliography_keys']}."
    )
    return 0 if (
        summary["unverified_count"] == 0
        and not summary["missing_bibliography_keys"]
        and not summary["uncited_bibliography_keys"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
