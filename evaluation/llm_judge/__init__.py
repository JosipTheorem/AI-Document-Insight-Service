"""Local evaluator configuration; its prompt lives alongside this file."""

from pathlib import Path

JUDGE_MODEL = "qwen3.5:9b-q4_K_M"
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        **{key: {"type": "boolean"} for key in ("retrieval_ok", "answer_ok")},
        "reason": {"type": "string"},
    },
    "required": ["retrieval_ok", "answer_ok", "reason"],
    "additionalProperties": False,
}


def load_prompt():
    return Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
