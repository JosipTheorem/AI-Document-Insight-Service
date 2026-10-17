"""Local HTTP backend shared by Streamlit and other clients."""

import io
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, Form, HTTPException, UploadFile
import pymupdf
from langchain_core.messages import ToolMessage
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from agent import build_agent
from config.retrieval import DOCUMENT_SESSION
from config.settings import PROJECT_ROOT
from retrieval import Index

logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 30 * 1024 * 1024
CITATION = re.compile(r"\[([^\]\n]+), str\. ([1-9]\d*)\]")


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
    app = FastAPI(title="AI Document Insight Service")
    sessions, registry_lock, ingestion_lock = {}, Lock(), Lock()
    with index.connection() as db:
        db.execute("CREATE TABLE IF NOT EXISTS api_sessions (id TEXT PRIMARY KEY)")

    def get_session(session_id):
        key = str(session_id)
        with registry_lock:
            with index.connection() as db:
                if not db.execute("SELECT 1 FROM api_sessions WHERE id=?", (key,)).fetchone():
                    raise HTTPException(404, "Sesija ne postoji. Započni novi razgovor.")
            if key not in sessions:
                sessions[key] = Session(build_agent(key, index))
            return sessions[key]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/sessions", status_code=201)
    def new_session(request: SessionRequest):
        session_id = str(uuid4())
        # Only the published fixture hashes may be shared with new conversations.
        if request.include_examples:
            manifest = PROJECT_ROOT / "files" / "sources.json"
            items = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
            hashes = [item["sha256"] for item in items if item["file"].startswith(("pdf/", "png/"))]
            index.copy_documents(DOCUMENT_SESSION, session_id, hashes)
        with index.connection() as db:
            db.execute("INSERT INTO api_sessions VALUES(?)", (session_id,))
        return {"session_id": session_id, "documents": index.documents(session_id)}

    @app.get("/documents")
    def documents(session_id: UUID):
        get_session(session_id)
        return {"documents": index.documents(str(session_id))}

    @app.post("/upload", response_model=UploadResponse)
    def upload(session_id: Annotated[UUID, Form()], file: UploadFile):
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
                return {**result, "session_id": session_id, "pages": doc["pages"], "chunks": doc["chunks"]}
        except ValueError as error:
            logger.warning("Document rejected: %s", error)
            raise HTTPException(422, "Dokument nije moguće obraditi. Provjeri sadržaj i ograničenja dokumenta.") from error
        except Exception as error:
            logger.exception("Document ingestion failed")
            raise HTTPException(503, "Obrada nije uspjela. Provjeri Ollamu i pokušaj ponovno.") from error

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest):
        session_id = str(request.session_id)
        session = get_session(session_id)
        try:
            with session.lock:
                result = session.agent.invoke(
                    {"messages": [{"role": "user", "content": request.message}]},
                    config={"configurable": {"thread_id": session_id}, "recursion_limit": 20},
                )
                answer = result["messages"][-1].text
                evidence = {source for message in result["messages"] if isinstance(message, ToolMessage)
                            and message.name == "search_documents" for source in CITATION.findall(message.text)}
                sources = [Source(filename=name, page_number=int(page)) for name, page in
                           dict.fromkeys(CITATION.findall(answer)) if (name, page) in evidence]
                return AskResponse(session_id=request.session_id, answer=answer, sources=sources)
        except Exception as error:
            logger.exception("Agent request failed")
            raise HTTPException(503, "Odgovor nije uspio. Provjeri Ollamu i pokušaj ponovno.") from error

    return app


app = create_app()
