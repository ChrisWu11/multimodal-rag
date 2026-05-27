import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.container import get_container  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the local RAG system from the command line.")
    parser.add_argument("question")
    parser.add_argument("--modality", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-openai", action="store_true")
    args = parser.parse_args()

    response = get_container().pipeline.answer(
        question=args.question,
        top_k=args.top_k,
        modality=args.modality,
        use_openai=not args.no_openai,
    )
    print(response.answer)
    print("\nEvidence:")
    for idx, item in enumerate(response.evidence, start=1):
        print(f"[{idx}] {item.title} score={item.score} source={item.source_path}")


if __name__ == "__main__":
    main()
