import sys
import time
import re
import json

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langgraph.checkpoint.memory import InMemorySaver

from config.models import get_model
from config.settings import MAX_SESSION_MESSAGES, SESSION_LIMIT_MESSAGE, THREAD_CONFIG
from config.retrieval import DOCUMENT_SESSION
from before_model import check_input
from prompts import load_prompt
from tools import get_weather, make_search_documents
from app_logging import bind_session, log_context, log_event


@wrap_model_call
def finish_document_reply(request, handler):
    """Bound search retries; use Ollama JSON decoding for an explicitly requested JSON reply."""
    start = next((i for i in range(len(request.messages) - 1, -1, -1)
                  if request.messages[i].type == "human"), None)
    if start is None:
        return handler(request)
    results = [m for m in request.messages[start + 1:]
               if m.type == "tool" and m.name == "search_documents"]
    overrides = {}
    if len(results) >= 2 or any(m.text.startswith(("No passages found", "Document search failed")) for m in results):
        overrides["tools"] = []
    json_requested = bool(re.search(r"\bJSON\b", request.messages[start].text, re.I))
    final_document_reply = request.messages[-1].type == "tool" and bool(results)
    if final_document_reply:
        citations = list(dict.fromkeys(re.findall(r"\[[^\]\n]+, str\. [1-9]\d*\]", "\n".join(m.text for m in results))))
        # Use the requested legal QA contract only when those fields were explicitly asked for.
        fields = ["answer", "article", "source"] if json_requested else ["answer", "sources"]
        explicit_contract = not json_requested or all(f'"{key}"' in request.messages[start].text for key in fields)
        schema = {"type": "object", "properties": {key: {"type": "string"} for key in fields},
                  "required": fields, "additionalProperties": False} if explicit_contract else "json"
        if explicit_contract:
            if json_requested:
                schema["properties"]["source"]["enum"] = ["", *citations]
            else:
                schema["properties"]["sources"] = {
                    "type": "array", "items": {"type": "string", "enum": citations or [""]}}
            if "article" in fields:
                articles = []
                for result in results:
                    for passage in result.text.split("\n\n"):
                        article = re.search(r"Članak (\d+)\.", passage)
                        clause = re.search(r"\((\d+)\)", passage)
                        if article:
                            articles.append(article[1] + (f" ({clause[1]})" if clause else ""))
                schema["properties"]["article"]["enum"] = list(dict.fromkeys(["", *articles]))
        overrides.update(model=request.model.model_copy(update={"format": schema}), tools=[])
    response = handler(request.override(**overrides))
    if final_document_reply:
        try:
            message = response.result[0]
            value = json.loads(message.text)
            if not value["answer"].strip():
                value["answer"] = "U vraćenim odlomcima nemam dovoljno podataka za pouzdan odgovor."
                if "source" in value:
                    value["source"] = ""
                if "sources" in value:
                    value["sources"] = []
                if "article" in value:
                    value["article"] = ""
            text = json.dumps(value, ensure_ascii=False) if json_requested else (
                value["answer"] + ("\n\n" + " ".join(value["sources"]) if value["sources"] else ""))
            response.result[0] = message.model_copy(update={"content": text})
        except (ValueError, KeyError, TypeError):
            pass  # Non-Ollama test models may return ordinary text.
    return response


def build_agent(document_session=DOCUMENT_SESSION, index=None):
    return create_agent(
        model=get_model(),
        tools=[get_weather, make_search_documents(document_session, index)],
        system_prompt=load_prompt(),
        middleware=[check_input, finish_document_reply],
        checkpointer=InMemorySaver(),
    )


def main() -> None:
    # Redirected Windows output can default to cp1252, which cannot print emoji.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    agent = build_agent()
    messages_used = 0
    print("Agent started. Type 'exit' or 'quit' to leave.")

    try:
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            if not user_input:
                continue

            # Send only the new message; memory keeps earlier turns in this process.
            messages_used += 1
            with log_context("terminal"):
                bind_session(THREAD_CONFIG["configurable"]["thread_id"])
                started = time.perf_counter()
                try:
                    result = agent.invoke(
                        {"messages": [{"role": "user", "content": user_input}]},
                        config=THREAD_CONFIG,
                    )
                    print(f"Agent: {result['messages'][-1].text}")
                    log_event("terminal_request", status_code=200,  #logging
                              duration_ms=round((time.perf_counter() - started) * 1000, 3))
                except Exception as error:
                    log_event("terminal_request", level="ERROR", status_code=503,  #logging
                              duration_ms=round((time.perf_counter() - started) * 1000, 3), error_type=type(error).__name__)
                    print("Odgovor nije uspio. Provjeri Ollamu i pokušaj ponovno.")
            if messages_used >= MAX_SESSION_MESSAGES:
                print(SESSION_LIMIT_MESSAGE)
                break
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
