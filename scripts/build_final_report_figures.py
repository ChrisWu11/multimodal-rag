#!/usr/bin/env python3
"""Build report figures from immutable evaluation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


RED = "#8C1D18"
GOLD = "#C99A18"
TEAL = "#177E89"
INK = "#24201E"
GREY = "#6C6560"
PALE = "#F5F1EC"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, path: Path):
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture(path: Path):
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.axis("off")
    boxes = [
        (0.02, 0.59, 0.18, 0.25, "Private corpus", "PDFs + metadata"),
        (0.27, 0.59, 0.18, 0.25, "Parse & chunk", "page / section / DOI"),
        (0.52, 0.59, 0.18, 0.25, "Index", "FTS + SBERT vectors"),
        (0.02, 0.12, 0.18, 0.25, "Question + image", "text or visual summary"),
        (0.27, 0.12, 0.18, 0.25, "Hybrid retrieval", "RRF top-k candidates"),
        (0.52, 0.12, 0.18, 0.25, "CrossEncoder", "pairwise reranking"),
        (0.77, 0.12, 0.20, 0.25, "Grounded answer", "Gemini + citations"),
    ]
    for x, y, w, h, title, sub in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=1.4, edgecolor=RED, facecolor=PALE,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
                fontsize=12, weight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=9.5, color=GREY)
    arrows = [
        ((0.20, 0.715), (0.27, 0.715)), ((0.45, 0.715), (0.52, 0.715)),
        ((0.11, 0.59), (0.11, 0.37)), ((0.20, 0.245), (0.27, 0.245)),
        ((0.45, 0.245), (0.52, 0.245)), ((0.70, 0.245), (0.77, 0.245)),
        ((0.61, 0.59), (0.38, 0.37)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.4, color=INK))
    ax.text(0.02, 0.96, "Auditable multimodal RAG architecture", fontsize=18,
            weight="bold", color=INK)
    ax.text(0.02, 0.90, "Offline knowledge preparation and query-time evidence grounding",
            fontsize=10.5, color=GREY)
    save(fig, path)


def interval_errors(
    analysis: dict | None,
    *,
    section: str,
    mode: str,
    metric: str,
) -> tuple[float, float] | None:
    if not analysis:
        return None
    if section == "retrieval":
        interval = (
            analysis.get("retrieval", {})
            .get("confidence_intervals", {})
            .get(mode, {})
            .get(metric)
        )
    else:
        interval = (
            analysis.get(section, {})
            .get("summaries", {})
            .get(mode, {})
            .get(metric)
        )
    if not interval:
        return None
    estimate = float(interval["estimate"])
    return (
        max(0.0, estimate - float(interval["ci_low"])),
        max(0.0, float(interval["ci_high"]) - estimate),
    )


def interval_estimate(
    analysis: dict | None,
    *,
    section: str,
    mode: str,
    metric: str,
) -> float | None:
    if not analysis:
        return None
    interval = (
        analysis.get(section, {})
        .get("summaries", {})
        .get(mode, {})
        .get(metric)
    )
    return float(interval["estimate"]) if interval and "estimate" in interval else None


def retrieval(summary_path: Path, path: Path, analysis: dict | None):
    rows = load(summary_path)["summaries"]
    names = {
        "fts_bm25": "Lexical",
        "st_dense": "Dense",
        "st_hybrid_rrf": "Hybrid RRF",
        "rerank_fts_top30": "Lexical + rerank",
        "rerank_dense_top30": "Dense + rerank",
        "rerank_hybrid_rrf_top30": "Hybrid + rerank",
    }
    rows = [row for row in rows if row["mode"] in names]
    labels = [names[row["mode"]] for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for offset, key, label, color in [
        (-width, "mrr", "MRR", RED),
        (0, "recall_at_5", "Recall@5", GOLD),
        (width, "ndcg_at_10", "nDCG@10", TEAL),
    ]:
        errors = [
            interval_errors(
                analysis,
                section="retrieval",
                mode=row["mode"],
                metric=key,
            )
            for row in rows
        ]
        yerr = None
        if all(error is not None for error in errors):
            yerr = np.array(errors).T
        bars = ax.bar(
            x + offset,
            [row[key] for row in rows],
            width,
            label=label,
            color=color,
            yerr=yerr,
            capsize=2.5 if yerr is not None else 0,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0, 0.90)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval performance (50 reviewed silver questions)", loc="left",
                 fontsize=15, weight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    save(fig, path)


def chunks(summary_path: Path, path: Path):
    rows = load(summary_path)["summaries"]
    fig, ax = plt.subplots(figsize=(9.8, 5.3))
    annotation_style = {
        "w350_o50": {"xytext": (-10, 8), "ha": "right"},
        "w650_o90": {"xytext": (10, 8), "ha": "left"},
        "w650_o0": {"xytext": (-10, 8), "ha": "right"},
        "w900_o150": {"xytext": (8, 8), "ha": "left"},
    }
    for row, color in zip(rows, [RED, GOLD, TEAL, "#625A9C"]):
        ax.scatter(row["chunk_count"], row["recall_at_5"], s=180, color=color,
                   edgecolor="white", linewidth=1.5, zorder=3)
        style = annotation_style[row["configuration"]]
        ax.annotate(
            row["configuration"].replace("w", "").replace("_o", " words / ") + " overlap",
            (row["chunk_count"], row["recall_at_5"]),
            xytext=style["xytext"], textcoords="offset points", fontsize=9,
            ha=style["ha"],
        )
    ax.set_xlabel("Number of indexed chunks")
    ax.set_ylabel("Recall@5")
    ax.set_ylim(0.50, 0.78)
    ax.set_xlim(3150, 7350)
    ax.set_title("Chunk granularity: early recall versus index size", loc="left",
                 fontsize=15, weight="bold")
    ax.grid(alpha=0.22)
    save(fig, path)


def score_chart(
    summary_path: Path,
    path: Path,
    kind: str,
    analysis: dict | None,
):
    rows = load(summary_path)["summaries"]
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    x = np.arange(len(rows))
    labels = [row["mode"].replace("_", " ") for row in rows]
    if kind == "generation":
        keys = [
            ("mean_correctness_0_2", "correctness", "Correctness", RED),
            ("mean_faithfulness_0_2", "faithfulness", "Faithfulness", GOLD),
            ("mean_completeness_0_2", "completeness", "Completeness", TEAL),
        ]
        title = "Generation quality by evidence condition"
        ylabel = "Mean judge score (0-2)"
        ylim = (0, 2.25)
        analysis_section = "generation"
    else:
        keys = [
            (
                "mean_caption_coverage_0_2",
                "caption_coverage",
                "Caption coverage",
                RED,
            ),
            (
                "mean_citation_quality_0_2",
                "citation_quality",
                "Citation quality",
                GOLD,
            ),
            (
                "mean_response_structure_0_2",
                "response_structure",
                "Response structure",
                TEAL,
            ),
        ]
        title = "Exploratory image-QA quality"
        ylabel = "Mean judge score (0-2)"
        ylim = (0, 2.25)
        analysis_section = "image_qa"
    width = 0.24
    for index, (key, analysis_key, label, color) in enumerate(keys):
        errors = [
            interval_errors(
                analysis,
                section=analysis_section,
                mode=row["mode"],
                metric=analysis_key,
            )
            for row in rows
        ]
        yerr = None
        if all(error is not None for error in errors):
            yerr = np.array(errors).T
        values = [
            interval_estimate(
                analysis,
                section=analysis_section,
                mode=row["mode"],
                metric=analysis_key,
            )
            or float(row[key])
            for row in rows
        ]
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=label,
            color=color,
            yerr=yerr,
            capsize=3 if yerr is not None else 0,
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
        )
        for bar_index, (bar, value) in enumerate(zip(bars, values)):
            upper = errors[bar_index][1] if errors[bar_index] is not None else 0.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + upper + 0.025 + index * 0.025,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x, labels)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=15, weight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.margins(x=0.08)
    save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--generation", type=Path)
    parser.add_argument("--image-qa", type=Path)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    analysis = load(args.analysis) if args.analysis else None
    architecture(args.output / "figure-1-architecture.png")
    retrieval(args.retrieval, args.output / "figure-2-retrieval.png", analysis)
    chunks(args.chunks, args.output / "figure-3-chunk-ablation.png")
    if args.generation:
        score_chart(
            args.generation,
            args.output / "figure-4-generation.png",
            "generation",
            analysis,
        )
    if args.image_qa:
        score_chart(
            args.image_qa,
            args.output / "figure-5-image-qa.png",
            "image",
            analysis,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
