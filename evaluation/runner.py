"""Document QA evaluation using the real agent, search tool and local 9B judge."""

import hashlib
import json
import time
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama

from agent import build_agent
from config import DOCUMENT_SESSION, EMBEDDING_MODEL, OLLAMA_URL, model_name
from retrieval import Index
from evaluation.llm_judge import JUDGE_MODEL, JUDGE_SCHEMA, Verdict, load_prompt
from app_logging import log_context
from tools.search import CITATION

FORMAT_REQUEST = (
    '\nOdgovori isključivo jednim JSON objektom s točno tri tekstualna polja: '
    '"article", "answer", "source". U answer kopiraj cijelu relevantnu rečenicu '
    'iz odlomka, sa svim uvjetima i alternativama. U article prepiši stvarni broj '
    'članka i stavka iz ISTOG odlomka, a u source citat "[naziv_datoteke, str. 1]". '
    'Ako odgovor nije u dokumentima, to jasno reci i ostavi article i source praznima. '
    'Bez Markdown oznaka i bez dodatnog teksta.'
)


def load_dataset(all_cases=False):
    """One frozen dataset; the notebook uses its five selected cases."""
    dataset = json.loads(Path(__file__).with_name("dataset").joinpath("questions.json").read_text(encoding="utf-8"))
    if not all_cases:
        dataset["cases"] = [case for case in dataset["cases"] if case["id"] in dataset["notebook_cases"]]
    return dataset


def valid_format(answer):
    try:
        value = json.loads(answer)
        return (
            isinstance(value, dict)
            and set(value) == {"article", "answer", "source"}
            and all(isinstance(item, str) for item in value.values())
            and bool(value["answer"].strip())
            and (bool(CITATION.fullmatch(value["source"]))
                 or value["article"] == value["source"] == "")
        )
    except (ValueError, TypeError):
        return False


def validate_references(dataset, index):
    # Refuse stale gold chunks; never regenerate references from search results.
    with index.connection() as db:
        for case in dataset["cases"]:
            gold = case["gold_chunk"]
            if gold is None:
                assert not case["answerable"]
                continue
            assert hashlib.sha256(gold["text"].encode()).hexdigest() == gold["sha256"]
            assert db.execute(
                "SELECT 1 FROM chunks c JOIN documents d ON d.id=c.document_id "
                "WHERE d.session_id=? AND d.filename=? AND c.page_number=? AND c.text=?",
                (DOCUMENT_SESSION, gold["filename"], gold["page"], gold["text"]),
            ).fetchone(), f"Missing/stale gold chunk: {case['id']}. Re-ingest files/pdf and files/png."


def new_report():
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "agent_model": model_name, "judge_model": JUDGE_MODEL,
        "embedding_model": EMBEDDING_MODEL, "session": DOCUMENT_SESSION,
        "cases": [],
    }


def make_judge():
    return ChatOllama(model=JUDGE_MODEL, base_url=OLLAMA_URL, temperature=0, reasoning=False,
                     num_ctx=8192, format=JUDGE_SCHEMA)


def score_case(case, result, error, judge):
    row = {"id": case["id"], "question": case["question"], "answerable": case["answerable"],
           "queries": [], "passages": [], "answer": "", "error": error, "judge_error": None}
    if result:
        for message in result["messages"]:
            if isinstance(message, AIMessage):
                row["queries"].extend(call["args"] for call in message.tool_calls
                                      if call["name"] == "search_documents")
            if isinstance(message, ToolMessage) and message.name == "search_documents":
                row["passages"].append(message.text)
        row["answer"] = result["messages"][-1].text
    row["tool_called"] = bool(row["queries"])
    gold = case["gold_chunk"]
    expected_passage = f"[{gold['filename']}, str. {gold['page']}]\n{gold['text']}" if gold else None
    row["exact_gold_hit"] = any(expected_passage in text for text in row["passages"]) if gold else None
    row["gold_text_hit"] = any(gold["text"] in text for text in row["passages"]) if gold else None
    row["strict_json"] = valid_format(row["answer"])
    parsed = json.loads(row["answer"]) if row["strict_json"] else {}
    row["abstained"] = parsed.get("article") == parsed.get("source") == ""
    payload = {"question": case["question"], "reference_answer": case["expected_answer"],
               "answerable": case["answerable"], "gold_chunk": gold,
               "actual_tool_results": row["passages"], "final_answer": row["answer"],
               "python_format_ok": row["strict_json"]}
    try:
        row["judge"] = Verdict.model_validate_json(judge.invoke([
            ("system", load_prompt()), ("human", json.dumps(payload, ensure_ascii=False))
        ]).text).model_dump()
    except Exception as judge_error:
        row["judge_error"] = str(judge_error)
        row["judge"] = {"retrieval_ok": False,
                        "answer_ok": False, "reason": "Judge failed; see judge_error."}
    return row


def run_cases(dataset, output_path):
    index = Index()
    validate_references(dataset, index)
    agent, judge, report = build_agent(index=index), make_judge(), new_report()
    for case in dataset["cases"]:
        print(f"Running {case['id']}...", flush=True)
        started = time.perf_counter()
        result, error = None, None
        try:
            with log_context("evaluation"):
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": case["question"] + FORMAT_REQUEST}]},
                    config={"configurable": {"thread_id": f"eval-{uuid4()}"}, "recursion_limit": 20},
                )
        except Exception as failure:
            error = str(failure)
        row = score_case(case, result, error, judge)
        row["seconds"] = round(time.perf_counter() - started, 1)
        report["cases"].append(row)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  gold={row['exact_gold_hit']}, JSON={row['strict_json']}, "
              f"answer={row['judge']['answer_ok']} ({row['seconds']}s)", flush=True)
    return report
