# Local evaluation

Knowledge files: two Croatian legal TEST EXCERPTS, supplied as two native PDFs,
two PNGs and two image-only PDFs. Provenance and permission are preserved in
`../files/ORIGIN.md` and `../files/SOURCES.md`. These are deliberately simplified
documents, not official consolidated laws or verified current legal advice.

## Run the five-question notebook

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r evaluation/requirements.txt
python -m ipykernel install --user --name abysalto --display-name "Python (abysalto)"
python -m jupyterlab evaluation/evaluation.ipynb
```

Choose **Python (abysalto)** and **Run All**. Ollama needs the installed 2B agent,
9B judge and embedding model. Index the documents first: `../files/SOURCES.md`.

- `dataset/questions.json`: five frozen questions, exact chunks and reference answers.
- `dataset/all_questions.json`: all 13 original questions, including one unanswerable case.
- `dataset/references/questions.json`: original reference file, copied verbatim from the ZIP.
- `llm_judge/prompt.txt`: independent 9B evidence/factual judging instructions.
- `llm_judge/__init__.py`: judge model and structured output schema.
- `runner.py`: actual 2B agent/tool calls, Python format validation and 9B judging.
- `results/results.json`: five observed traces and verdicts from the executed notebook.
- `results/documents.png`, `results/metrics.png`: notebook figures.

The requested JSON fields are `article`, `answer`, `source`. References never go
to the conversational agent. Gold chunks are frozen from native extraction and
the supplied answers BEFORE search, not selected from search hits. References
are validated against the active index before the notebook runs.

`exact_gold_hit` requires the named native-PDF citation and full source chunk.
`gold_text_hit` requires the exact reference text, accepting paired file formats.
`strict_json` uses Python to check exact keys, string values and citation shape;
Markdown fences fail. Formatting is not guessed by the judge. For a missing
answer, article and source may both be empty. `abstained` records that structure
with a nonempty answer; it does not by itself prove factual correctness.

## Run API and security simulations

```powershell
python -m evaluation.simulations
```

Models, HTTP endpoints, uploads, OCR, embeddings, SQLite and tools are real.
An observer only records graph messages; it does not fake model/tool outputs.
The runner uses an isolated temporary index and checks:

- All six uploads and the expected native/OCR extraction method.
- Chats in native-only, PNG-only and scanned-PDF-only sessions.
- All 13 questions through the API, with the independent 9B judge.
- Empty-session isolation, duplicate and invalid uploads, and the 10-message limit.

Results: `results/simulations.json`. The three natural-language chat checks
were also verified through the running HTTP server after the final formatter
change; this is recorded in `natural_chat_rechecked_at` and those check rows.
The 2B guard is checked against `../tests/security_examples.json` plus all 13
legal QA requests, with results in `results/security.json`.

Security prompt: `../before_model/prompt.txt`; validated output schema:
`../before_model/schema.py`. Neutral legal questions are allowed. In the saved
run one classifier-manipulation example still bypasses this simple LLM guard.
This finite set of probes is not a security guarantee.

In the saved 13-question run the 9B judge rejects the unanswerable case despite
the agent explicitly refusing to answer for lack of evidence. Both the actual
reply and unmodified judge verdict are retained and shown in the notebook.

Notebook logs: `data/evaluation-events.txt`. Simulation logs:
`data/simulation-events.txt`. Normal API/terminal logs remain in
`app_logging/logs/application.txt`, so isolated test traffic does not inflate the
normal dashboard. Live UI/terminal smoke requests are normal logged requests.

## Retrieval and interpretation

Hybrid search uses BM25, cosine similarity and weighted RRF (BM25 weight 1,
vector weight 2, k=60). Semantic matching gets more influence for Croatian word
inflections. This setting is tuned on this small corpus, not an independent
benchmark. Legal articles/subsections stay together in chunks. Identical
extracted passages across PDF/PNG/scan variants are returned only once so they
do not crowd out other evidence. All formats are still tested independently.

Requested article/citation fields are constrained to identifiers present in
tool results. A blank generated answer becomes an explicit abstention. Normal
chat replies can cite multiple sources. Paired documents and related questions
are not independent samples; these runs are smoke checks, not general quality
or throughput benchmarks. Inspect actual answers and judge reasons.
