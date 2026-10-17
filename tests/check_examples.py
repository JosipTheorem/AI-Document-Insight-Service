"""Opt-in real Ollama checks after indexing the three provided examples in session demo.

Run: python -m tests.check_examples
These are retrieval smoke checks, not a comprehensive quality benchmark.
"""

import json
import sys
from pathlib import Path

from retrieval import Index


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    service = Index()
    checks = json.loads(Path(__file__).with_name("example_queries.json").read_text(encoding="utf-8"))
    failures = []
    for check in checks:
        hits = service.search(check["query"], session_id="demo", top_k=3)
        correct = any(
            hit["filename"] == check["filename"] and hit["page_number"] == check["page"]
            and check["evidence"].casefold() in hit["text"].casefold()
            for hit in hits
        )
        print(f"{'PASS' if correct else 'FAIL'}: {check['query']}", flush=True)
        for i, hit in enumerate(hits, 1):
            print(f"  {i}. {hit['filename']}, page {hit['page_number']}, "
                  f"keyword={hit['lexical_rank']}, vector={hit['vector_rank']}, RRF={hit['rrf_score']:.6f}", flush=True)
        if not correct:
            failures.append(check["query"])
    if failures:
        raise AssertionError(f"Expected evidence missing from top 3: {failures}")
    print(f"All {len(checks)} example retrieval checks passed.")


if __name__ == "__main__":
    main()
