import sys

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from config.models import get_model
from config.settings import THREAD_CONFIG
from prompts import load_system_prompt
from tools import get_weather, search_documents


def build_agent():
    return create_agent(
        model=get_model(),
        tools=[get_weather, search_documents],
        system_prompt=load_system_prompt(),
        checkpointer=InMemorySaver(),
    )


def main() -> None:
    # Redirected Windows output can default to cp1252, which cannot print emoji.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    agent = build_agent()
    print("Agent started. Type 'exit' or 'quit' to leave.")

    try:
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            if not user_input:
                continue

            # Send only the new message; memory keeps earlier turns in this process.
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=THREAD_CONFIG,
            )
            print(f"Agent: {result['messages'][-1].text}")
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
