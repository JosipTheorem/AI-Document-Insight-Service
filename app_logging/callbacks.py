"""Collect provider-reported tokens once per completed LLM call, not per chunk."""

import time
from threading import Lock

from langchain_core.callbacks import BaseCallbackHandler

from . import current_context, log_event


class UsageLogger(BaseCallbackHandler):
    run_inline = True

    def __init__(self, purpose, model):
        self.purpose, self.model = purpose, model
        self.runs, self.lock = {}, Lock()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        with self.lock:
            self.runs[run_id] = (time.perf_counter(), current_context())

    def finish(self, run_id):
        with self.lock:
            started, context = self.runs.pop(run_id, (time.perf_counter(), current_context()))
        return {**context, "run_id": str(run_id), "purpose": self.purpose, "model": self.model,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3)}

    def on_llm_end(self, response, *, run_id, **kwargs):
        call = self.finish(run_id)
        # ChatOllama returns the combined message for the complete invocation.
        message = response.generations[0][0].message
        metadata = message.response_metadata or {}
        usage = message.usage_metadata or {}
        incoming = usage.get("input_tokens", metadata.get("prompt_eval_count"))
        outgoing = usage.get("output_tokens", metadata.get("eval_count"))
        known = type(incoming) is int and type(outgoing) is int and incoming >= 0 and outgoing >= 0
        finish_reason = metadata.get("done_reason")
        log_event("llm_completed", level="WARNING" if finish_reason == "length" else "INFO",  #logging
                  **call, usage_available=known, finish_reason=finish_reason,
                  has_text=bool(message.text.strip()), tool_calls=len(message.tool_calls),
                  input_tokens=incoming if known else None, output_tokens=outgoing if known else None,
                  generation_ms=(metadata.get("eval_duration") or 0) / 1_000_000,
                  prompt_ms=(metadata.get("prompt_eval_duration") or 0) / 1_000_000,
                  load_ms=(metadata.get("load_duration") or 0) / 1_000_000)

    def on_llm_error(self, error, *, run_id, **kwargs):
        log_event("llm_error", level="ERROR", **self.finish(run_id), error_type=type(error).__name__)  #logging
