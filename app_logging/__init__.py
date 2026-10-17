"""Small JSON-line event log; no external logging service required."""

import json
import os
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from hashlib import sha256
from pathlib import Path
from threading import Lock
from uuid import uuid4

_context = ContextVar("log_context", default={})
_write_lock = Lock()


def log_path():
    default = Path(__file__).parent / "logs" / "application.txt"
    return Path(os.getenv("APP_LOG_FILE", default))


def current_context():
    return dict(_context.get())


def bind_session(session_id):
    # Session IDs are access keys, so only a short fingerprint goes into logs.
    _context.set({**_context.get(), "session_ref": sha256(str(session_id).encode()).hexdigest()[:16]})


@contextmanager
def log_context(interface, request_id=None):
    token = _context.set({"interface": interface, "request_id": request_id or str(uuid4())})
    try:
        yield
    finally:
        _context.reset(token)


def log_event(event, level="INFO", **fields):
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event_id": str(uuid4()),
              "level": level, "event": event, "interface": "direct", **current_context(), **fields}
    line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
    try:
        with _write_lock:
            path = log_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as output:
                output.write(line)  #logging
    except OSError:
        # A full/unwritable log disk must not break the user's request.
        sys.stderr.write("Application event log could not be written.\n")  #logging


def timed(event):
    """Measure a pipeline stage without recording its input or output contents."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            started = time.perf_counter()
            try:
                result = function(*args, **kwargs)
            except Exception as error:
                log_event("phase_error", level="ERROR", phase=event, error_type=type(error).__name__,  #logging
                          duration_ms=round((time.perf_counter() - started) * 1000, 3))
                raise
            log_event(event, duration_ms=round((time.perf_counter() - started) * 1000, 3))  #logging
            return result
        return wrapped
    return decorate
