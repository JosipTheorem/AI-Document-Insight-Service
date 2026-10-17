"""Five end-to-end cases through the same agent used by terminal and Streamlit."""

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama

from agent import build_agent
from config.models import model_name
from config.retrieval import DOCUMENT_SESSION, EMBEDDING_MODEL
from retrieval import Index
from evaluation.llm_judge import JUDGE_MODEL, JUDGE_SCHEMA, load_prompt

FORMAT_REQUEST = (
    '\nReturn ONLY a JSON object with exactly three string fields: '
    '"invoice_number", "answer", "source". Include the requested value and its units '
    'in answer. Source must be a citation like "[filename.pdf, str. 1]". No markdown.'
)


def valid_format(answer):
    try:
        value = json.loads(answer)
        return (
            isinstance(value, dict)
            and set(value) == {"invoice_number", "answer", "source"}
            and all(isinstance(item, str) for item in value.values())
            and bool(re.fullmatch(r"\[[^\]\n]+, str\. [1-9]\d*\]", value["source"]))
        )
    except (ValueError, TypeError):
        return False


def run_cases(dataset, output_path):
    # Refuse stale gold chunks; never regenerate references from search results.
    with Index().connection() as db:
        for case in dataset["cases"]:
            gold = case["gold_chunk"]
            assert hashlib.sha256(gold["text"].encode()).hexdigest() == gold["sha256"]
            assert db.execute(
                "SELECT 1 FROM chunks c JOIN documents d ON d.id=c.document_id "
                "WHERE d.session_id=? AND d.filename=? AND c.page_number=? AND c.text=?",
                (DOCUMENT_SESSION, gold["filename"], gold["page"], gold["text"]),
            ).fetchone(), f"Missing/stale gold chunk: {case['id']}. Re-ingest files/pdf and files/png."

    agent = build_agent()
    judge = ChatOllama(model=JUDGE_MODEL, temperature=0, reasoning=False,
                       num_ctx=8192, format=JUDGE_SCHEMA)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "agent_model": model_name, "judge_model": JUDGE_MODEL,
        "embedding_model": EMBEDDING_MODEL, "session": DOCUMENT_SESSION,
        "cases": [],
    }
    for case in dataset["cases"]:
        print(f"Running {case['id']}...", flush=True)
        started = time.perf_counter()
        row = {"id": case["id"], "question": case["question"], "queries": [],
               "passages": [], "answer": "", "error": None, "judge_error": None}
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": case["question"] + FORMAT_REQUEST}]},
                config={"configurable": {"thread_id": f"eval-{uuid4()}"}, "recursion_limit": 20},
            )
            for message in result["messages"]:
                if isinstance(message, AIMessage):
                    row["queries"].extend(call["args"] for call in message.tool_calls
                                          if call["name"] == "search_documents")
                if isinstance(message, ToolMessage) and message.name == "search_documents":
                    row["passages"].append(message.text)
            row["answer"] = result["messages"][-1].text
        except Exception as error:
            row["error"] = str(error)
        row["tool_called"] = bool(row["queries"])
        gold = case["gold_chunk"]
        expected_passage = f"[{gold['filename']}, str. {gold['page']}]\n{gold['text']}"
        row["exact_gold_hit"] = any(expected_passage in text for text in row["passages"])
        row["strict_json"] = valid_format(row["answer"])
        payload = {"question": case["question"], "reference_answer": case["expected_answer"],
                   "gold_chunk": gold, "actual_tool_results": row["passages"],
                   "final_answer": row["answer"]}
        try:
            verdict = json.loads(judge.invoke([
                ("system", load_prompt()), ("human", json.dumps(payload, ensure_ascii=False))
            ]).text)
            assert set(verdict) == set(JUDGE_SCHEMA["required"])
            assert all(type(verdict[key]) is bool for key in ("retrieval_ok", "format_ok", "answer_ok"))
            row["judge"] = verdict
        except Exception as error:
            row["judge_error"] = str(error)
            row["judge"] = {"retrieval_ok": False, "format_ok": False,
                            "answer_ok": False, "reason": "Judge failed; see judge_error."}
        row["seconds"] = round(time.perf_counter() - started, 1)
        report["cases"].append(row)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  gold={row['exact_gold_hit']}, JSON={row['strict_json']}, "
              f"answer={row['judge']['answer_ok']} ({row['seconds']}s)", flush=True)
    return report
