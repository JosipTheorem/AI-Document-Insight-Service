import sys
import time
import re
import json

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langgraph.checkpoint.memory import InMemorySaver

from config import DOCUMENT_SESSION, MAX_SESSION_MESSAGES, PROJECT_ROOT, SESSION_LIMIT_MESSAGE, get_model
from before_model import check_input
from tools import get_weather, make_search_documents
from tools.search import CITATION
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
        passages = "\n\n".join(m.text for m in results)
        citations = list(dict.fromkeys(match[0] for match in CITATION.finditer(passages)))
        # Use the requested legal QA contract only when those fields were explicitly asked for.
        properties = {"answer": {"type": "string"}}
        schema = "json"
        if not json_requested or all(f'"{key}"' in request.messages[start].text for key in ("answer", "article", "source")):
            if json_requested:
                articles = [""]
                for passage in passages.split("\n\n"):
                    article = re.search(r"Članak (\d+)\.", passage)
                    clause = re.search(r"\((\d+)\)", passage)
                    if article:
                        articles.append(article[1] + (f" ({clause[1]})" if clause else ""))
                properties["article"] = {"type": "string", "enum": list(dict.fromkeys(articles))}
                properties["source"] = {"type": "string", "enum": ["", *citations]}
            else:
                properties["sources"] = {"type": "array", "items": {"type": "string", "enum": citations or [""]}}
            schema = {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
        overrides.update(model=request.model.model_copy(update={"format": schema}), tools=[])
    response = handler(request.override(**overrides))
    if final_document_reply:
        try:
            message = response.result[0]
            value = json.loads(message.text)
            if not value["answer"].strip():
                value["answer"] = "U vraćenim odlomcima nemam dovoljno podataka za pouzdan odgovor."
                for key, empty in (("source", ""), ("sources", []), ("article", "")):
                    if key in value:
                        value[key] = empty
            sources = dict.fromkeys(value.get("sources", []))
            extra = " ".join(source for source in sources if source and source not in value["answer"])
            text = json.dumps(value, ensure_ascii=False) if json_requested else value["answer"] + ("\n\n" + extra if extra else "")
            response.result[0] = message.model_copy(update={"content": text})
        except (ValueError, KeyError, TypeError):
            pass  # Non-Ollama test models may return ordinary text.
    return response


def build_agent(document_session=DOCUMENT_SESSION, index=None):
    return create_agent(
        model=get_model(),
        tools=[get_weather, make_search_documents(document_session, index)],
        system_prompt=(PROJECT_ROOT / "prompts/system.txt").read_text(encoding="utf-8").strip(),
        middleware=[check_input, finish_document_reply],
        checkpointer=InMemorySaver(),
    )


def main() -> None:
    # Redirected Windows output can default to cp1252, which cannot print emoji.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    agent = build_agent()
    thread = {"configurable": {"thread_id": "terminal-chat"}}
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
                bind_session("terminal-chat")
                started = time.perf_counter()
                try:
                    result = agent.invoke(
                        {"messages": [{"role": "user", "content": user_input}]},
                        config=thread,
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
