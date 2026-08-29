import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rejudge_generation import recover_raw_judgments, valid_judge  # noqa: E402


def complete_judge() -> dict[str, object]:
    return {
        "correctness": 2,
        "faithfulness": 1,
        "completeness": 2,
        "factual_claims": 3,
        "unsupported_claims": 1,
        "claims_with_citations": 2,
        "supported_cited_claims": 2,
        "abstained": False,
        "brief_reason": "Supported overall.",
    }


def test_valid_judge_rejects_incomplete_or_out_of_range_values() -> None:
    assert valid_judge(complete_judge())
    assert not valid_judge({**complete_judge(), "correctness": 3})
    assert not valid_judge({**complete_judge(), "unsupported_claims": -1})
    assert not valid_judge({**complete_judge(), "abstained": "false"})


def test_recover_raw_judgments_keeps_only_valid_expected_rows(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw-batches.jsonl"
    response = [
        {"task_key": "q1::dense_rag::1", "judge": complete_judge()},
        {
            "task_key": "q2::dense_rag::1",
            "judge": {**complete_judge(), "faithfulness": 4},
        },
        {"task_key": "unexpected::dense_rag::1", "judge": complete_judge()},
    ]
    raw_path.write_text(
        json.dumps(
            {
                "judge_model": "gemini-test",
                "task_keys": ["q1::dense_rag::1", "q2::dense_rag::1"],
                "latency_ms": 120,
                "response": json.dumps(response),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    recovered = recover_raw_judgments(
        raw_path,
        valid_task_keys={"q1::dense_rag::1", "q2::dense_rag::1"},
        judge_model="gemini-test",
    )

    assert len(recovered) == 1
    assert recovered[0]["id"] == "q1"
    assert recovered[0]["judge_latency_ms"] == 60
