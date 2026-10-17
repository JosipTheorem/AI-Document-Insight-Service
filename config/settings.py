from pathlib import Path

from dotenv import load_dotenv


# Resolve paths from this file, regardless of the terminal's working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

THREAD_CONFIG = {"configurable": {"thread_id": "terminal-chat"}}
