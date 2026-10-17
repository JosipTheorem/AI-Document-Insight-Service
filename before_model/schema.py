"""Validated output schema sent to Ollama for the security classifier."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class SecurityDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["OK", "NOT_OK"]
