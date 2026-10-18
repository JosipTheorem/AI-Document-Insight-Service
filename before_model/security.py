"""Check new user input before the model; skip subsequent tool-result turns."""

from pathlib import Path

from langchain.agents.middleware import before_model
from langchain_core.messages import AIMessage, RemoveMessage

from config import get_model
from app_logging import log_event
from .schema import SecurityDecision

PROMPT = Path(__file__).with_name("prompt.txt")
guard = get_model(purpose="security").model_copy(update={"reasoning": False}).with_structured_output(
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
        log_event("security_blocked", level="WARNING")  #logging
        refusal = "Ne mogu pomoći s uvredljivim porukama ili zaobilaženjem pravila. Slobodno postavi drugo pitanje."
    except Exception as error:
        log_event("security_error", level="ERROR", error_type=type(error).__name__)  #logging
        refusal = "Poruku trenutačno ne mogu provjeriti. Pokušaj ponovno."
    return {"messages": [RemoveMessage(id=message.id), AIMessage(refusal)], "jump_to": "end"}
