import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.container import get_container  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a file or directory into the local RAG store.")
    parser.add_argument("path", help="File or directory to ingest.")
    parser.add_argument("--modality", default="text", help="text, ultrasound, thermal, or unknown.")
    args = parser.parse_args()

    container = get_container()
    target = Path(args.path)
    files = [target] if target.is_file() else [p for p in target.rglob("*") if p.is_file()]

    for file_path in files:
        if file_path.name.startswith("."):
            continue
        data = file_path.read_bytes()
        response = container.ingestor.ingest_file(
            filename=file_path.name,
            data=data,
            title=file_path.stem,
            modality=args.modality,
            metadata={"source": "cli", "path": str(file_path)},
        )
        print(f"{response.document_id} indexed {response.chunks_indexed} chunks from {file_path}")


if __name__ == "__main__":
    main()
