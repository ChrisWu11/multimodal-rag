#!/usr/bin/env python3
"""Rejudge an existing generation run with one fixed Gemini model."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402

import final_report_eval as evaluation  # type: ignore[import-not-found]  # noqa: E402


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "final-report-evaluation"
ORDINAL_METRICS = ("correctness", "faithfulness", "completeness")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_raw_batch(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def valid_judge(judge: Any) -> bool:
    if not isinstance(judge, dict) or not isinstance(judge.get("abstained"), bool):
        return False
    for metric in ORDINAL_METRICS:
        value = judge.get(metric)
        if not isinstance(value, int) or isinstance(value, bool) or value not in {0, 1, 2}:
            return False
    for metric in (
        "factual_claims",
        "unsupported_claims",
        "claims_with_citations",
        "supported_cited_claims",
    ):
        value = judge.get(metric)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return False
    return True


def recover_raw_judgments(
    raw_path: Path,
    *,
    valid_task_keys: set[str],
    judge_model: str,
) -> list[dict[str, Any]]:
    """Recover valid rows from a batch response that omitted other task keys."""
    if not raw_path.exists():
        return []
    recovered: dict[tuple[str, str, int], dict[str, Any]] = {}
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                batch = json.loads(line)
                if batch.get("judge_model") != judge_model:
                    continue
                expected = set(batch.get("task_keys") or []) & valid_task_keys
                parsed = evaluation.parse_json_array(str(batch.get("response") or ""))
                latency = float(batch.get("latency_ms") or 0)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            for row in parsed:
                task_key = str(row.get("task_key") or "")
                judge = row.get("judge")
                if task_key not in expected or not valid_judge(judge):
                    continue
                try:
                    item_id, mode, repeat_text = task_key.rsplit("::", 2)
                    repeat = int(repeat_text)
                except (TypeError, ValueError):
                    continue
                recovered[(item_id, mode, repeat)] = {
                    "id": item_id,
                    "mode": mode,
                    "repeat": repeat,
                    "judge_model": judge_model,
                    "judge": judge,
                    "judge_batch_latency_ms": round(latency, 2),
                    "judge_latency_ms": round(
                        latency / max(len(batch.get("task_keys") or []), 1),
                        2,
                    ),
                }
    return list(recovered.values())


def aggregate_summary(
    answers: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    answer_by_key = {
        (str(row["id"]), str(row["mode"])): row
        for row in answers
    }
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in judgments:
        by_key.setdefault((str(row["id"]), str(row["mode"])), []).append(row["judge"])

    rows = []
    for mode in ("llm_only", "dense_rag", "hybrid_reranked_rag"):
        answerable_keys = [
            key
            for key, answer in answer_by_key.items()
            if key[1] == mode and answer["answerable"]
        ]
        unanswerable_keys = [
            key
            for key, answer in answer_by_key.items()
            if key[1] == mode and not answer["answerable"]
        ]
        answerable_judges = [
            judge
            for key in answerable_keys
            for judge in by_key.get(key, [])
        ]
        all_keys = answerable_keys + unanswerable_keys
        all_judges = [judge for key in all_keys for judge in by_key.get(key, [])]
        unanswerable_judges = [
            judge
            for key in unanswerable_keys
            for judge in by_key.get(key, [])
        ]
        total_claims = sum(int(row.get("factual_claims") or 0) for row in all_judges)
        unsupported = sum(int(row.get("unsupported_claims") or 0) for row in all_judges)
        cited = sum(
            int(row.get("claims_with_citations") or 0)
            for key in answerable_keys
            for row in by_key.get(key, [])
        )
        supported_cited = sum(
            int(row.get("supported_cited_claims") or 0)
            for key in answerable_keys
            for row in by_key.get(key, [])
        )
        rows.append(
            {
                "mode": mode,
                "answerable_count": len(answerable_keys),
                "unanswerable_count": len(unanswerable_keys),
                **{
                    f"mean_{metric}_0_2": round(
                        fmean(float(row.get(metric) or 0) for row in answerable_judges),
                        3,
                    )
                    for metric in ORDINAL_METRICS
                },
                "unsupported_claim_rate_all_cases": round(
                    unsupported / max(total_claims, 1),
                    3,
                ),
                "citation_precision_answerable": (
                    None
                    if mode == "llm_only"
                    else round(supported_cited / max(cited, 1), 3)
                ),
                "invalid_citation_count": sum(
                    len(answer_by_key[key].get("invalid_citations") or [])
                    for key in all_keys
                ),
                "abstention_accuracy_unanswerable": round(
                    sum(bool(row.get("abstained")) for row in unanswerable_judges)
                    / max(len(unanswerable_judges), 1),
                    3,
                ),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--judge-model", default="gemini-3.6-flash")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--seconds-between-batches", type=float, default=8.0)
    args = parser.parse_args()

    source_run = args.source_run.expanduser().resolve()
    answers_path = source_run / "answers.json"
    manifest_path = source_run / "manifest.json"
    if not answers_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("The source run must contain answers.json and manifest.json.")
    answers = json.loads(answers_path.read_text(encoding="utf-8"))

    if args.resume_dir:
        run_dir = args.resume_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = evaluation.create_run_dir(
            args.output_root.expanduser().resolve(),
            "generation-fixed-judge",
        )
    output_answers = run_dir / "answers.json"
    output_manifest = run_dir / "manifest.json"
    if not output_answers.exists():
        shutil.copy2(answers_path, output_answers)
    if not output_manifest.exists():
        shutil.copy2(manifest_path, output_manifest)

    settings = get_settings()
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = ChatGoogleGenerativeAI(
        model=args.judge_model,
        api_key=settings.gemini_api_key,
        temperature=0,
    )
    judge_system = (
        "You are evaluating a research RAG answer. Evaluate every task independently and return "
        "only the requested JSON array. Use integers 0, 1 or 2 for correctness, faithfulness and "
        "completeness. Correctness, faithfulness and completeness describe answerable questions. "
        "For unanswerable questions, abstained is true only when the answer refuses to invent an "
        "answer. Count atomic factual claims, unsupported claims, claims with citations and cited "
        "claims actually supported by the supplied evidence. A syntactically valid citation is "
        "not supported unless its evidence entails the claim. Do not reward fluency."
    )

    tasks = [
        (answer, repeat)
        for answer in answers
        for repeat in range(1, args.repeats + 1)
    ]
    valid_task_keys = {
        f"{answer['id']}::{answer['mode']}::{repeat}"
        for answer, repeat in tasks
    }
    partial_path = run_dir / "judgments.partial.json"
    raw_path = run_dir / "raw-batches.jsonl"
    existing = (
        json.loads(partial_path.read_text(encoding="utf-8"))
        if partial_path.exists()
        else []
    )
    recovered = recover_raw_judgments(
        raw_path,
        valid_task_keys=valid_task_keys,
        judge_model=args.judge_model,
    )
    existing_by_key = {
        (str(row["id"]), str(row["mode"]), int(row["repeat"])): row
        for row in recovered
    }
    existing_by_key.update(
        {
            (str(row["id"]), str(row["mode"]), int(row["repeat"])): row
            for row in existing
        }
    )
    existing = sorted(
        existing_by_key.values(),
        key=lambda row: (str(row["id"]), str(row["mode"]), int(row["repeat"])),
    )
    if recovered:
        evaluation.write_json(partial_path, existing)
    existing_keys = {
        (str(row["id"]), str(row["mode"]), int(row["repeat"]))
        for row in existing
    }
    remaining = [
        task
        for task in tasks
        if (str(task[0]["id"]), str(task[0]["mode"]), int(task[1]))
        not in existing_keys
    ]

    def judge_batch(batch: list[tuple[dict[str, Any], int]]) -> list[dict[str, Any]]:
        payload = []
        for item, repeat in batch:
            payload.append(
                {
                    "task_key": f"{item['id']}::{item['mode']}::{repeat}",
                    "question": item["question"],
                    "answerable": item["answerable"],
                    "reference_answer": item["gold_answer"],
                    "retrieved_evidence": evaluation.evidence_context(item["evidence"]),
                    "candidate_answer": item["answer"],
                }
            )
        request = {
            "tasks": payload,
            "required_output": (
                "A JSON array. Each item contains task_key and judge. judge contains correctness, "
                "faithfulness and completeness (integers 0-2); factual_claims, unsupported_claims, "
                "claims_with_citations and supported_cited_claims (integers >=0); abstained "
                "(boolean); brief_reason (at most 40 words)."
            ),
        }
        response_text, latency = evaluation.invoke_with_retry(
            model,
            [
                SystemMessage(content=judge_system),
                HumanMessage(content=json.dumps(request, ensure_ascii=False)),
            ],
        )
        append_raw_batch(
            raw_path,
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "judge_model": args.judge_model,
                "task_keys": [item["task_key"] for item in payload],
                "latency_ms": round(latency, 2),
                "response": response_text,
            },
        )
        parsed = evaluation.parse_json_array(response_text)
        expected = {item["task_key"] for item in payload}
        rows = []
        for row in parsed:
            task_key = str(row.get("task_key"))
            if task_key not in expected:
                continue
            item_id, mode, repeat = task_key.rsplit("::", 2)
            rows.append(
                {
                    "id": item_id,
                    "mode": mode,
                    "repeat": int(repeat),
                    "judge_model": args.judge_model,
                    "judge": row["judge"],
                    "judge_batch_latency_ms": round(latency, 2),
                    "judge_latency_ms": round(latency / len(batch), 2),
                }
            )
        returned = {
            f"{row['id']}::{row['mode']}::{row['repeat']}"
            for row in rows
        }
        missing = sorted(expected - returned)
        if missing:
            raise ValueError(f"Judge omitted task keys: {missing}")
        return rows

    new_rows: list[dict[str, Any]] = []
    batches = [
        remaining[start : start + args.batch_size]
        for start in range(0, len(remaining), args.batch_size)
    ]
    for index, batch in enumerate(batches, start=1):
        rows = judge_batch(batch)
        new_rows.extend(rows)
        combined = existing + new_rows
        combined.sort(key=lambda row: (str(row["id"]), str(row["mode"]), int(row["repeat"])))
        evaluation.write_json(partial_path, combined)
        print(
            f"Judged batch {index}/{len(batches)}; "
            f"rows {len(combined)}/{len(tasks)}",
            flush=True,
        )
        if index < len(batches):
            time.sleep(args.seconds_between_batches)

    judgments = existing + new_rows
    judgments.sort(key=lambda row: (str(row["id"]), str(row["mode"]), int(row["repeat"])))
    models = Counter(str(row.get("judge_model")) for row in judgments)
    if set(models) != {args.judge_model}:
        raise RuntimeError(f"Fixed-judge run contains unexpected models: {models}")
    evaluation.write_json(run_dir / "judgments.json", judgments)
    summary = {
        "task": "generation_fixed_model_rejudge",
        "git_sha": evaluation.git_sha(),
        "source_run": str(source_run),
        "source_answers_sha256": sha256(answers_path),
        "judge_model": args.judge_model,
        "judge_repeats": args.repeats,
        "judgment_count": len(judgments),
        "scoring_scope": {
            "quality": "Answerable questions only.",
            "unsupported_claims": "All cases.",
            "abstention": "Designed unanswerable questions only.",
            "citation_precision": "Answerable grounded modes; not applicable to LLM only.",
        },
        "summaries": aggregate_summary(answers, judgments),
    }
    evaluation.write_json(run_dir / "summary.json", summary)
    evaluation.write_csv(run_dir / "summary.csv", summary["summaries"])
    evaluation.write_json(
        run_dir / "run-config.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "judge_model": args.judge_model,
            "repeats": args.repeats,
            "batch_size": args.batch_size,
            "seconds_between_batches": args.seconds_between_batches,
            "source_run": str(source_run),
            "source_manifest_sha256": sha256(manifest_path),
            "source_answers_sha256": sha256(answers_path),
            "no_model_fallback": True,
        },
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
