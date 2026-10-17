from pathlib import Path


def load_system_prompt() -> str:
    """Read the system prompt relative to this package, not the working directory."""
    return Path(__file__).with_name("system.txt").read_text(encoding="utf-8").strip()
