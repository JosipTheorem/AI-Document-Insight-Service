# Local evaluation

Open `evaluation.ipynb` to see the saved run, five sample images, metrics and answers.

To run it again from the project root (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r evaluation/requirements.txt
python -m ipykernel install --user --name abysalto --display-name "Python (abysalto)"
python -m jupyterlab evaluation/evaluation.ipynb
```

Choose **Python (abysalto)** and **Run All**. Ollama must be running with the 2B
agent, 9B judge and embedding models installed. Index the documents first using
the command in `../files/SOURCES.md` if using a fresh checkout.

- `dataset/questions.json`: five frozen questions, exact gold chunks and reference answers.
- `dataset/references/`: original field values provided by the invoice author.
- `llm_judge/prompt.txt`: instructions for the independent 9B judge.
- `llm_judge/__init__.py`: judge model and output schema.
- `runner.py`: real agent/tool execution and independent local judge.
- `results/results.json`: observed queries, passages, answers, verdicts and errors.
- `results/baseline.json`: original run before fixing the security prompt.
- `results/documents.png`, `results/metrics.png`: notebook figures.
- `results/security.json`: real local security-model checks.

The notebook stays at `evaluation/evaluation.ipynb`. The actual knowledge files
live in `files/pdf/` and `files/png/`; evaluation uses that same small corpus,
without duplicating it. `data/` is the generated, Git-ignored SQLite index,
embeddings and ingestion snapshots, not a second evaluation dataset.

Security runs through `before_model/security.py`. Its prompt is
`before_model/prompt.txt`; `before_model/schema.py` defines a validated Pydantic
output with only `decision: OK | NOT_OK`, sent to Ollama as a JSON schema.
Only new human messages are classified; tool responses skip that check.

Reference answers never go to the conversational agent. The judge accepts
equivalent evidence from a paired PDF/PNG; the exact-gold metric deliberately
requires the named source and full text. Invalid judge responses are recorded
as failures with an error, rather than silently retried or awarded points.
The JSON format check measures structure, not factual accuracy. Five synthetic
cases are a smoke check and do not establish general model quality.

Samples: SalorWorks, Synthetic Shopify Invoice Test Pack v1.0.0, CC BY 4.0.
See `../files/SOURCES.md` for attribution and pinned source revision.
