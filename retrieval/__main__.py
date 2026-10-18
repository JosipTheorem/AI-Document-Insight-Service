"""Terminal commands: ingest, search, list and inspect."""

import argparse
import json
import sys
from pathlib import Path

from config import DATABASE, DOCUMENT_SESSION
from retrieval import Index


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Local document ingestion and hybrid search")
    parser.add_argument("--session", default=DOCUMENT_SESSION)
    parser.add_argument("--data-dir", type=Path, default=DATABASE.parent)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("files", nargs="+")
    ingest.add_argument("--force", action="store_true")
    search = commands.add_parser("search")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--document", action="append")
    commands.add_parser("list")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("document_id")
    inspect.add_argument("--page", type=int, default=1)
    args = parser.parse_args()
    index = Index(args.data_dir / "index.sqlite3")
    try:
        if args.command == "ingest":
            result = [index.ingest(path, args.session, args.force) for path in args.files]
        elif args.command == "search":
            result = index.search(args.query, args.session, args.top_k, args.document)
        elif args.command == "inspect":
            result = index.inspect(args.document_id, args.page, args.session)
        else:
            result = index.documents(args.session)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
