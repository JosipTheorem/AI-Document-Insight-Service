"""Check new user input before the model; skip subsequent tool-result turns."""

from pathlib import Path

from langchain.agents.middleware import before_model
from langchain_core.messages import AIMessage, RemoveMessage

from config.models import get_model
from .schema import SecurityDecision

PROMPT = Path(__file__).with_name("prompt.txt")
guard = get_model().model_copy(update={"reasoning": False}).with_structured_output(
    SecurityDecision, method="json_schema",
)


@before_model(can_jump_to=["end"])
def check_input(state, runtime):
    message = state["messages"][-1]
    if message.type != "human":
        return None
    try:
        reply = guard.invoke([("system", PROMPT.read_text(encoding="utf-8")), ("human", message.text)])
        if reply.decision == "OK":
            return None
        refusal = "Ne mogu pomoći s uvredljivim porukama ili zaobilaženjem pravila. Slobodno postavi drugo pitanje."
    except Exception:
        refusal = "Poruku trenutačno ne mogu provjeriti. Pokušaj ponovno."
    return {"messages": [RemoveMessage(id=message.id), AIMessage(refusal)], "jump_to": "end"}
