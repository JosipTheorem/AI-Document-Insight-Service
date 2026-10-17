from pathlib import Path


def load_prompt(name: str = "system.txt") -> str:
    """Read a prompt relative to this package, not the working directory."""
    return Path(__file__).with_name(name).read_text(encoding="utf-8").strip()
