from pathlib import Path

from dotenv import load_dotenv


# Resolve paths from this file, regardless of the terminal's working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

THREAD_CONFIG = {"configurable": {"thread_id": "terminal-chat"}}
MAX_SESSION_MESSAGES = 10
EXAMPLE_DATASET = "croatian-legal-v1"
SESSION_LIMIT_MESSAGE = "Iskoristio si svih 10 poruka u ovoj sesiji. Pokreni novi razgovor za nastavak."
