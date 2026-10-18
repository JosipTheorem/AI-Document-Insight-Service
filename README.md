# AI Document Insight Service

A small local document QA application. Upload a PDF, PNG or JPEG, then ask questions
about its contents. It includes a FastAPI backend, a Streamlit chat interface and
a separate monitoring dashboard. Models run through Ollama; no cloud API key is needed.

## Start with Docker

Install and start Docker Desktop, then run:

```shell
git clone https://github.com/JosipTheorem/AI-Document-Insight-Service.git
cd AI-Document-Insight-Service
docker compose up -d --build
```

For an NVIDIA GPU, use this last command instead:

```shell
docker compose -f compose.yaml -f compose.gpu.yaml up -d --build
```

The first build downloads dependencies and OCR models. Startup downloads missing
Ollama models and indexes the example documents. This can take a while, especially
on CPU. Later starts reuse the models and database.

Open:

- Chat: http://127.0.0.1:8501
- API documentation: http://127.0.0.1:8000/docs
- Monitoring: http://127.0.0.1:8502

The chat starts with the included examples. For your own document, select a file
and click **Obradi dokument**. Once processing finishes, ask a question using Enter
or the send button. Previous conversations are available in the sidebar.
Each session allows ten requests, including refused or failed attempts.

Example questions are in [data/files/questions.txt](data/files/questions.txt).
The examples are simplified Croatian legal test excerpts: two topics represented
as text PDFs, scanned PDFs and PNGs. They are not current legal advice.
Source and reuse information is in [data/files/ORIGIN.md](data/files/ORIGIN.md).

To stop the services while keeping their data:

```shell
docker compose stop
```

Docker stores the database and uploaded document copies in the `document_data`
volume. Logs are written to the host's `app_logging/logs/application.txt`.
An optional `.env`, based on [.env.example](.env.example), can change host ports or
reuse an existing Ollama model directory. In GPU mode, include the same two `-f`
options in subsequent Compose startup commands.

## Run manually

Use Python 3.13 and install/start Ollama on your computer. From the project root,
run these PowerShell commands:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m deployment.bootstrap
```

On Linux/macOS, activate the environment with `source .venv/bin/activate` instead.
The bootstrap command downloads missing application models and indexes the examples.
Redis is optional for manual use: an unavailable cache falls back to calculating
embeddings. To disable it explicitly, set `REDIS_URL=` in `.env`. To start just Redis
with Docker, run `docker compose up -d redis`.

Start the API:

```shell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

In another terminal with the virtual environment activated, start the chat:

```shell
python -m streamlit run app.py
```

Optionally, start monitoring in a third terminal:

```shell
python -m streamlit run monitoring/dashboard.py --server.port 8502
```

Choose either the full Docker setup or the manual setup; they use the same ports.
Manual runs store their database in `data/runtime/index.sqlite3`, separately from
Docker's volume. Terminal-only chat is also available with `python agent.py`;
its conversation memory lasts for that process.

## API example

The Swagger UI at `/docs` can run these requests interactively. This PowerShell
example creates an empty session, uploads a document and asks about it:

```powershell
$base = 'http://127.0.0.1:8000'
$session = Invoke-RestMethod -Method Post -Uri "$base/sessions" `
    -ContentType 'application/json' -Body '{"include_examples":false}'
$id = $session.session_id

curl.exe -X POST "$base/upload" -F "session_id=$id" `
    -F 'file=@data/files/pdf/kazneni_zakon_izvadak.pdf'

$body = @{session_id=$id; message='What is the prison sentence for basic theft?'} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$base/ask" `
    -ContentType 'application/json' -Body $body
```

Example upload response (IDs vary):

```json
{
  "session_id": "<session-id>",
  "status": "indexed",
  "document_id": "<document-id>",
  "filename": "kazneni_zakon_izvadak.pdf",
  "pages": 1,
  "chunks": 10,
  "warnings": []
}
```

Example answer response (wording may vary):

```json
{
  "session_id": "<session-id>",
  "answer": "For basic theft, the test excerpt states a prison sentence of up to three years.\n\n[kazneni_zakon_izvadak.pdf, str. 1]",
  "sources": [{"filename": "kazneni_zakon_izvadak.pdf", "page_number": 1}],
  "messages_used": 1,
  "messages_remaining": 9,
  "message_limit": 10,
  "notice": null
}
```

`GET /sessions` lists saved conversations, `GET /sessions/{id}` restores one,
and `GET /documents?session_id={id}` lists its documents. Uploads accept one file
per request: PDF, PNG or JPEG, up to 30 MB. PDFs are limited to 200 pages.

## How it works

1. **Extract:** [PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html)
   handles page layout, columns and tables. Its
   [RapidOCR plugin](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/ocr-plugins.html)
   reads scanned text. Images use the same pipeline after conversion to PDF.
2. **Prepare:** normalize text, preserve article/subsection boundaries where
   possible, and split longer sections into 220-word chunks with 35-word overlap.
3. **Store:** SQLite holds documents, extracted pages, chunks, embedding vectors
   and API conversation history. Redis caches document and query embeddings.
4. **Search:** combine SQLite FTS5/BM25 keyword results and NumPy cosine similarity
   using weighted reciprocal rank fusion. RRF uses positions because the two raw
   scores have different scales. Current weights are 1 for BM25 and 2 for vectors,
   with a rank constant of 60. Return up to three distinct passages to the agent.
5. **Answer:** a LangChain agent calls the search tool and answers with file/page
   citations. A `before_model` check screens new input using a structured decision.

Models:

| Purpose | Model |
| --- | --- |
| Chat and input security check | `qwen3.5:2b-q4_K_M` |
| Embeddings | `qwen3-embedding:0.6b` |
| Evaluation judge | `qwen3.5:9b-q4_K_M` |

The 2B model keeps local inference practical; the larger model is used only for
evaluation. SQLite keeps storage simple, while Redis avoids repeated embedding
calculations. PyMuPDF4LLM and RapidOCR keep document processing local.

## Project map

| Location | Purpose |
| --- | --- |
| `app.py`, `api.py`, `agent.py` | UI, HTTP backend and agent |
| `config.py` | Shared settings and model setup |
| `retrieval/` | Extraction, chunking, embeddings, cache and hybrid search |
| `tools/search.py` | Session-scoped document search tool |
| `prompts/`, `before_model/` | Main prompt and input security check/prompt/schema |
| `data/files/` | Included PDFs, images, sample questions and attribution |
| `data/runtime/` | Generated local database and document copies; ignored by Git |
| `app_logging/`, `monitoring/` | Structured text logs and dashboard |
| `evaluation/` | Notebook, frozen dataset, judge prompt and saved results |
| `tests/`, `deployment/` | Regression checks and startup preparation |

## Tests and evaluation

Run regression tests without an Ollama server:

```shell
python -m unittest discover -s tests -v
```

They cover extraction, tables, columns, session isolation, uploads, conversation
restoration, limits, caching, logging and error handling. Model and embedding
calls are mocked. Real-model checks use the evaluation notebook and simulations.

For the Docker notebook, download the judge model and start Jupyter:

```shell
docker compose exec ollama ollama pull qwen3.5:9b-q4_K_M
docker compose --profile evaluation up -d --build evaluation
docker compose logs evaluation
```

Open the Jupyter URL/token shown in the logs at http://127.0.0.1:8888 and run
`evaluation.ipynb` from top to bottom. For manual use, pull the same judge model
with `ollama pull qwen3.5:9b-q4_K_M`, then run
`python -m jupyterlab evaluation/evaluation.ipynb`.

The notebook evaluates five selected questions against frozen expected passages
and answer requirements. Python checks retrieval/format; the local 9B judge
checks factual correctness. `python -m evaluation.simulations` runs the broader
13-question API evaluation and security probes. Results are saved under
`evaluation/results/`. Small evaluation sets and LLM judges have limitations;
the saved traces are there to inspect, not just the final scores.

In the [saved notebook run](evaluation/results/results.json), all five questions
retrieved the expected text and passed the format check; the judge accepted three
answers. The [broader API run](evaluation/results/simulations.json) accepted seven
of thirteen answers. These results show the current 2B model's limitations.

## Current limits

This is a local demo with shared conversations and no user authentication. The
ten-request quota is per session, not a production rate limiter. Input checks and
upload validation are present, but the LLM security check can be bypassed.

The 2B model can misinterpret a passage, choose an irrelevant search or produce
an incorrect answer despite retrieving the right chunk. OCR can also make errors.
RRF ranks available passages; it does not guarantee that a question is answerable.
Stored conversations provide history, but there is no cross-session semantic memory.

## Possible additions

- **Context compaction:** summarize older turns before the context window fills.
- **Long-term memory:** retain useful facts/preferences across conversations.
- **After-agent layer:** validate grounding, citations and the final answer before display.
- **CI/CD:** run tests on pull requests and deploy isolated preview environments automatically.
- **LLM-assisted chunk creation:** suggest semantic boundaries and metadata while preserving source text.
- **LLM-assisted retrieval:** rerank/check the relevance of returned passages before answering.
- Streaming answers in the UI.
- Conversation/document deletion, retention rules and session cleanup.
