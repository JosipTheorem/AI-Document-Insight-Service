"""Local evaluator configuration; its prompt lives alongside this file."""

from pathlib import Path
from pydantic import BaseModel, ConfigDict

JUDGE_MODEL = "qwen3.5:9b-q4_K_M"


class Verdict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    retrieval_ok: bool
    answer_ok: bool
    reason: str


JUDGE_SCHEMA = Verdict.model_json_schema()


def load_prompt():
    return Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
