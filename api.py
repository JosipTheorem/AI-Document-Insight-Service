"""Local HTTP backend shared by Streamlit and other clients."""

import io
import json
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
import pymupdf
from langchain_core.messages import ToolMessage, messages_from_dict, messages_to_dict
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from agent import build_agent
from config import DOCUMENT_SESSION, MAX_SESSION_MESSAGES, SESSION_LIMIT_MESSAGE, example_files
from retrieval import Index
from app_logging import bind_session, current_context, log_context, log_event
from tools.search import CITATION

MAX_UPLOAD_BYTES = 30 * 1024 * 1024


class SessionRequest(BaseModel):
    include_examples: bool = True


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: UUID
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class Source(BaseModel):
    filename: str
    page_number: int


class AskResponse(BaseModel):
    session_id: UUID
    answer: str
    sources: list[Source]
    messages_used: int
    messages_remaining: int
    message_limit: int = MAX_SESSION_MESSAGES
    notice: str | None = None


class UploadResponse(BaseModel):
    session_id: UUID
    status: Literal["indexed", "unchanged"]
    document_id: str
    filename: str
    pages: int
    chunks: int
    warnings: list[str] = Field(default_factory=list)


@dataclass
class Session:
    agent: Any
    lock: Any = field(default_factory=Lock)


def validate_upload(filename, payload):
    """Check the filename, size and actual file type before ingestion."""
    if not re.fullmatch(r'[^<>:"/\\|?*\x00-\x1f]{1,180}', filename) or filename.endswith((" ", ".")):
        raise HTTPException(400, "Neispravan naziv datoteke.")
    if Path(filename).stem.upper() in {"CON", "PRN", "AUX", "NUL", *[f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10)]}:
        raise HTTPException(400, "Neispravan naziv datoteke.")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg"}:
        raise HTTPException(415, "Podržani su PDF, PNG i JPEG.")
    if not payload:
        raise HTTPException(400, "Datoteka je prazna.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Najveća dopuštena veličina je 30 MB.")
    try:
        if suffix == ".pdf":
            if not payload.lstrip().startswith(b"%PDF-"):
                raise ValueError("Invalid PDF signature")
            with pymupdf.open(stream=payload, filetype="pdf") as pdf:
                if pdf.needs_pass or not 1 <= len(pdf) <= 200:
                    raise ValueError("Encrypted, empty or oversized PDF")
        else:
            with Image.open(io.BytesIO(payload)) as image:
                expected = "PNG" if suffix == ".png" else "JPEG"
                if image.format != expected or image.width * image.height > 40_000_000:
                    raise ValueError("Invalid image format or size")
                image.verify()
    except (ValueError, OSError, pymupdf.FileDataError, Image.DecompressionBombError) as error:
        raise HTTPException(400, "Datoteka nije ispravna ili ne odgovara nastavku.") from error


def create_app(index=None):
    index = index if index is not None else Index()

    @asynccontextmanager
    async def lifespan(app):
        log_event("application_started", interface="api")  #logging
        yield
        log_event("application_stopped", interface="api")  #logging

    app = FastAPI(title="AI Document Insight Service", lifespan=lifespan)
    sessions, registry_lock, ingestion_lock = {}, Lock(), Lock()
    with index.connection() as db:
        db.execute("CREATE TABLE IF NOT EXISTS api_sessions (id TEXT PRIMARY KEY)")
        columns = {row["name"] for row in db.execute("PRAGMA table_info(api_sessions)")}
        for name, definition in (("message_count", "INTEGER NOT NULL DEFAULT 0"), ("uploaded_file", "TEXT")):
            if name not in columns:
                db.execute(f"ALTER TABLE api_sessions ADD COLUMN {name} {definition}")
        db.execute("""CREATE TABLE IF NOT EXISTS conversation_turns (
            id INTEGER PRIMARY KEY, session_id TEXT NOT NULL REFERENCES api_sessions(id),
            question TEXT NOT NULL, answer TEXT NOT NULL, state_json TEXT NOT NULL)""")

    @app.middleware("http")
    async def record_request(request: Request, call_next):
        started, status = time.perf_counter(), 500
        with log_context("api"):
            request_id = current_context()["request_id"]
            try:
                response = await call_next(request)
                status = response.status_code
                response.headers["X-Request-ID"] = request_id
                return response
            except Exception as error:
                log_event("http_error", level="ERROR", error_type=type(error).__name__)  #logging
                raise
            finally:
                log_event("http_request", level="ERROR" if status >= 500 else "WARNING" if status >= 400 else "INFO",  #logging
                          method=request.method, path=request.url.path, status_code=status,
                          duration_ms=round((time.perf_counter() - started) * 1000, 3))

    def usage(session_id):
        with index.connection() as db:
            row = db.execute("SELECT message_count FROM api_sessions WHERE id=?", (str(session_id),)).fetchone()
        used = row["message_count"]
        return {"messages_used": used, "messages_remaining": max(0, MAX_SESSION_MESSAGES - used),
                "message_limit": MAX_SESSION_MESSAGES,
                "notice": SESSION_LIMIT_MESSAGE if used >= MAX_SESSION_MESSAGES else None}

    def load_agent(key):
        agent = build_agent(key, index)
        with index.connection() as db:
            saved = db.execute("SELECT state_json FROM conversation_turns WHERE session_id=? "
                               "ORDER BY id DESC LIMIT 1", (key,)).fetchone()
        if saved:
            agent.update_state({"configurable": {"thread_id": key}},
                               {"messages": messages_from_dict(json.loads(saved["state_json"]))})
        return agent

    def get_session(session_id):
        key = str(session_id)
        bind_session(key)
        with registry_lock:
            with index.connection() as db:
                if not db.execute("SELECT 1 FROM api_sessions WHERE id=?", (key,)).fetchone():
                    raise HTTPException(404, "Sesija ne postoji. Započni novi razgovor.")
            if key not in sessions:
                sessions[key] = Session(load_agent(key))
            return sessions[key]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/sessions")
    def list_sessions():
        # Local demo: conversations are shared by everyone using this instance.
        public_hashes = json.dumps([item["sha256"] for item in example_files()])
        with index.connection() as db:
            rows = db.execute("""SELECT s.id AS session_id,
                (SELECT question FROM conversation_turns WHERE session_id=s.id ORDER BY id LIMIT 1) AS title,
                COALESCE(s.uploaded_file, (SELECT filename FROM documents WHERE session_id=s.id AND content_hash NOT IN
                    (SELECT value FROM json_each(?)) ORDER BY created_at,id LIMIT 1)) AS uploaded_file
                FROM api_sessions s ORDER BY
                (SELECT MAX(id) FROM conversation_turns WHERE session_id=s.id) DESC, s.rowid DESC""",
                              (public_hashes,)).fetchall()
        return [{"session_id": row["session_id"],
                 "title": " ".join(row["title"].split())[:60] if row["title"] else f"Dokument: {row['uploaded_file']}"}
                for row in rows if row["title"] or row["uploaded_file"]]

    @app.get("/sessions/{session_id}")
    def conversation(session_id: UUID):
        get_session(session_id)
        with index.connection() as db:
            rows = db.execute("SELECT question, answer FROM conversation_turns WHERE session_id=? ORDER BY id",
                              (str(session_id),)).fetchall()
        messages = [message for row in rows for message in (
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"] or "Model tada nije dovršio odgovor. Možeš ponovno poslati pitanje."})]
        return {"session_id": str(session_id), "messages": messages,
                "documents": index.documents(str(session_id)), **usage(session_id)}

    @app.post("/sessions", status_code=201)
    def new_session(request: SessionRequest):
        session_id = str(uuid4())
        bind_session(session_id)
        # Only the published fixture hashes may be shared with new conversations.
        if request.include_examples:
            hashes = [item["sha256"] for item in example_files()]
            index.copy_documents(DOCUMENT_SESSION, session_id, hashes)
        with index.connection() as db:
            db.execute("INSERT INTO api_sessions(id) VALUES(?)", (session_id,))
        documents = index.documents(session_id)
        log_event("session_created", document_count=len(documents))  #logging
        return {"session_id": session_id, "documents": documents, **usage(session_id)}

    @app.get("/documents")
    def documents(session_id: UUID):
        get_session(session_id)
        return {"documents": index.documents(str(session_id)), **usage(session_id)}

    @app.post("/upload", response_model=UploadResponse)
    def upload(session_id: Annotated[UUID, Form()], file: UploadFile):
        started = time.perf_counter()
        session = get_session(session_id)
        try:
            payload = file.file.read(MAX_UPLOAD_BYTES + 1)
        finally:
            file.file.close()
        filename = file.filename or ""
        validate_upload(filename, payload)
        staging = index.path.parent / "uploads"
        staging.mkdir(exist_ok=True)
        try:
            with session.lock, ingestion_lock, TemporaryDirectory(dir=staging) as temporary:
                path = Path(temporary) / filename
                path.write_bytes(payload)
                result = index.ingest(path, session_id=str(session_id))
                doc = next(d for d in index.documents(str(session_id)) if d["id"] == result["document_id"])
                with index.connection() as db:
                    db.execute("UPDATE api_sessions SET uploaded_file=COALESCE(uploaded_file,?) WHERE id=?",
                               (filename, str(session_id)))
                log_event("upload_completed", status=result["status"], bytes=len(payload),  #logging
                          pages=doc["pages"], chunks=doc["chunks"], document_id=result["document_id"],
                          duration_ms=round((time.perf_counter() - started) * 1000, 3))
                return {**result, "session_id": session_id, "pages": doc["pages"], "chunks": doc["chunks"]}
        except ValueError as error:
            log_event("upload_rejected", level="WARNING", error_type=type(error).__name__)  #logging
            raise HTTPException(422, "Dokument nije moguće obraditi. Provjeri sadržaj i ograničenja dokumenta.") from error
        except Exception as error:
            log_event("upload_error", level="ERROR", error_type=type(error).__name__)  #logging
            raise HTTPException(503, "Obrada nije uspjela. Provjeri Ollamu i pokušaj ponovno.") from error

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest):
        session_id = str(request.session_id)
        session = get_session(session_id)
        try:
            with session.lock:
                # Reserve before calling either model; blocked/failed attempts also count.
                with index.connection() as db:
                    reserved = db.execute("UPDATE api_sessions SET message_count=message_count+1 "
                                          "WHERE id=? AND message_count<?", (session_id, MAX_SESSION_MESSAGES)).rowcount
                if not reserved:
                    log_event("session_limit_reached", level="WARNING")  #logging
                    raise HTTPException(429, SESSION_LIMIT_MESSAGE)
                try:
                    result = session.agent.invoke(
                        {"messages": [{"role": "user", "content": request.message}]},
                        config={"configurable": {"thread_id": session_id}, "recursion_limit": 20},
                    )
                    answer = result["messages"][-1].text
                    if not answer.strip():
                        raise RuntimeError("Agent returned an empty answer")
                    evidence = {source for message in result["messages"] if isinstance(message, ToolMessage)
                                and message.name == "search_documents" for source in CITATION.findall(message.text)}
                    sources = [Source(filename=name, page_number=int(page)) for name, page in
                               dict.fromkeys(CITATION.findall(answer)) if (name, page) in evidence]
                    with index.connection() as db:
                        db.execute("INSERT INTO conversation_turns(session_id, question, answer, state_json) VALUES(?,?,?,?)",
                                   (session_id, request.message, answer, json.dumps(messages_to_dict(result["messages"]))))
                    stats = usage(session_id)
                    log_event("agent_completed", sources=len(sources), **stats)  #logging
                    return AskResponse(session_id=request.session_id, answer=answer, sources=sources, **stats)
                except Exception:
                    # Discard unfinished graph steps before another request can use this session.
                    session.agent = load_agent(session_id)
                    raise
        except HTTPException:
            raise
        except Exception as error:
            log_event("agent_error", level="ERROR", error_type=type(error).__name__)  #logging
            raise HTTPException(503, "Odgovor nije uspio. Provjeri Ollamu i pokušaj ponovno.") from error

    return app


app = create_app()
