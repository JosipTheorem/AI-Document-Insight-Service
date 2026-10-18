"""One local index: ingestion, SQLite persistence, embeddings and hybrid search."""

import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from uuid import uuid4

import numpy as np
from ollama import Client

from config import (
    BATCH_SIZE, CANDIDATES, DATABASE, DOCUMENT_SESSION, EMBEDDING_MODEL,
    OLLAMA_URL, PIPELINE_ID, RRF_K, RRF_WEIGHTS,
)
from retrieval.documents import clean_text, extract, make_chunks
from retrieval.cache import EmbeddingCache
from app_logging import timed

SCHEMA = """
CREATE TABLE IF NOT EXISTS index_meta (id INTEGER PRIMARY KEY CHECK(id=1), model_id TEXT NOT NULL, dimension INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, filename TEXT NOT NULL,
    content_hash TEXT NOT NULL, stored_path TEXT NOT NULL, pipeline_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, UNIQUE(session_id, content_hash));
CREATE TABLE IF NOT EXISTS pages (
    document_id TEXT REFERENCES documents(id) ON DELETE CASCADE, page_number INTEGER,
    raw_text TEXT NOT NULL, clean_text TEXT NOT NULL, method TEXT NOT NULL,
    blocks_json TEXT NOT NULL, warnings_json TEXT NOT NULL, PRIMARY KEY(document_id,page_number));
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL, ordinal INTEGER NOT NULL, text TEXT NOT NULL,
    kind TEXT NOT NULL, embedding BLOB NOT NULL, UNIQUE(document_id,ordinal));
CREATE INDEX IF NOT EXISTS documents_session ON documents(session_id);
CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='unicode61 remove_diacritics 2');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid,text) VALUES(new.id,new.text); END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts,rowid,text) VALUES('delete',old.id,old.text); END;
"""
STOP_WORDS = set((
    "a an the is are was were be been of to in on at for from with and or how what which "
    "who when where why does do did can could should would please tell me about "
    "i u o na za iz od do se su je li te ili da kako koji koja koje tko što sta sto "
    "kada gdje mi reci molim prema s sa kao jer ali biti bio bila bili bilo bi bude "
    "ce će dok trebam treba trebaju može mogu možete ima imaju ovo ovaj ova ovdje "
    "taj ta to te tih tim tome tom također"
).split())


def keywords(query):
    terms = dict.fromkeys(re.findall(r"[^\W_]+", query.casefold()))
    return " OR ".join(f'"{t}"' for t in terms if t not in STOP_WORDS and (len(t) > 1 or t.isdigit()))


def rrf(*rankings, k=RRF_K, weights=None):
    scores = defaultdict(float)
    for ranking, weight in zip(rankings, weights or [1] * len(rankings), strict=True):
        for rank, chunk_id in enumerate(dict.fromkeys(ranking), 1):
            scores[chunk_id] += weight / (k + rank)
    return sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id)), scores


def normalize(values):
    vectors = np.asarray(values, dtype="<f4")
    if vectors.ndim != 2 or not vectors.shape[1] or not np.isfinite(vectors).all():
        raise ValueError("Invalid embedding vectors.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding vectors must not be zero.")
    return vectors / norms


class Index:
    def __init__(self, path=DATABASE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.client = Client(host=OLLAMA_URL, timeout=180)
        self.cache = EmbeddingCache()
        with self.connection() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @cached_property
    def model_id(self):
        for model in self.client.list().models:
            if model.model == EMBEDDING_MODEL:
                return f"{EMBEDDING_MODEL}@{model.digest}"
        raise ValueError(f"Run: ollama pull {EMBEDDING_MODEL}")

    @timed("embedding_completed")  #logging
    def embed(self, texts, query=False):
        if query:
            texts = [f"Instruct: Given a question, retrieve relevant document passages that answer the question\nQuery: {t}" for t in texts]
        keys = [self.cache.key(self.model_id, text, query) for text in texts]
        pending = dict(zip(keys, texts, strict=True))
        vectors = self.cache.get_many(list(pending))
        missing = [key for key in pending if key not in vectors]
        fresh = {}
        for start in range(0, len(missing), BATCH_SIZE):
            batch_keys = missing[start:start + BATCH_SIZE]
            batch = [pending[key] for key in batch_keys]
            response = self.client.embed(model=EMBEDDING_MODEL, input=batch, truncate=False, keep_alive="10m")
            computed = normalize(response.embeddings)
            if len(computed) != len(batch):
                raise ValueError("Embedding count does not match the input.")
            fresh.update(zip(batch_keys, computed, strict=True))
        vectors.update(fresh)
        result = normalize([vectors[key] for key in keys])
        self.cache.put_many(fresh)
        return result

    def ingest(self, path, session_id=DOCUMENT_SESSION, force=False):
        path = Path(path).expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg"}:
            raise ValueError("Expected a PDF, PNG or JPEG file.")
        if not session_id.strip() or path.stat().st_size > 30 * 1024 * 1024:
            raise ValueError("Session is required and files must be at most 30 MB.")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        with self.connection() as db:
            old = db.execute("SELECT * FROM documents WHERE session_id=? AND content_hash=?", (session_id, digest)).fetchone()
            meta = db.execute("SELECT * FROM index_meta").fetchone()
        if meta and meta["model_id"] != self.model_id:
            raise ValueError("Embedding model changed. Use a new index and re-ingest.")
        if old and old["pipeline_id"] == PIPELINE_ID and not force:
            return dict(status="unchanged", document_id=old["id"], filename=old["filename"])
        snapshot = self.path.parent / "documents" / f"{digest}{path.suffix.lower()}"
        snapshot.parent.mkdir(exist_ok=True)
        snapshot.write_bytes(payload)
        pages = extract(snapshot)
        chunks = make_chunks(pages)
        if not chunks:
            raise ValueError("No readable text found.")
        vectors = normalize(self.embed([chunk["text"] for chunk in chunks]))
        document_id = old["id"] if old else str(uuid4())
        with self.connection() as db:
            meta = db.execute("SELECT * FROM index_meta").fetchone()
            if meta and (meta["model_id"] != self.model_id or meta["dimension"] != vectors.shape[1]):
                raise ValueError("Embedding model/dimension differs from the index.")
            db.execute("INSERT OR IGNORE INTO index_meta VALUES(1,?,?)", (self.model_id, vectors.shape[1]))
            db.execute("DELETE FROM documents WHERE id=?", (document_id,))
            db.execute("INSERT INTO documents(id,session_id,filename,content_hash,stored_path,pipeline_id) VALUES(?,?,?,?,?,?)",
                       (document_id, session_id, path.name, digest, str(snapshot), PIPELINE_ID))
            db.executemany("INSERT INTO pages VALUES(?,?,?,?,?,?,?)", [
                (document_id, p["number"], p["raw"], clean_text("\n\n".join(text for _, text in p["blocks"])),
                 p["method"], json.dumps(p["blocks"], ensure_ascii=False), json.dumps(p["warnings"])) for p in pages])
            db.executemany("INSERT INTO chunks(document_id,page_number,ordinal,text,kind,embedding) VALUES(?,?,?,?,?,?)", [
                (document_id, c["page_number"], i, c["text"], c["kind"], v.astype("<f4").tobytes())
                for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))])
        return dict(status="indexed", document_id=document_id, filename=path.name, pages=len(pages), chunks=len(chunks),
                    warnings=[note for page in pages for note in page["warnings"]])

    @timed("search_completed")  #logging
    def search(self, query, session_id=DOCUMENT_SESSION, top_k=5, document_ids=None):
        query = query.strip()
        if not query or len(query) > 2000 or not 1 <= top_k <= 20 or not session_id.strip():
            raise ValueError("Expected a session, a question of 1-2000 characters and top_k of 1-20.")
        sql, args = "SELECT c.*,d.filename FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.session_id=?", [session_id]
        if document_ids is not None:
            sql += " AND d.id IN (SELECT value FROM json_each(?))"
            args.append(json.dumps(document_ids))
        with self.connection() as db:
            rows = {row["id"]: dict(row) for row in db.execute(sql + " ORDER BY c.id", args)}
            meta = db.execute("SELECT * FROM index_meta").fetchone()
        if not rows:
            return []
        if meta["model_id"] != self.model_id:
            raise ValueError("Embedding model changed. Re-ingest into a new index.")
        vector = normalize(self.embed([query], query=True))[0]
        if len(vector) != meta["dimension"]:
            raise ValueError("Question embedding dimension differs from the index.")
        matrix = np.vstack([np.frombuffer(row["embedding"], dtype="<f4") for row in rows.values()])
        cosine = dict(zip(rows, np.clip(matrix @ vector, -1, 1).tolist()))
        count = max(CANDIDATES, top_k)
        semantic = sorted(rows, key=lambda i: (-cosine[i], i))[:count]
        expression = keywords(query)
        with self.connection() as db:
            lexical = list(db.execute("""SELECT rowid,bm25(chunks_fts) FROM chunks_fts
                WHERE chunks_fts MATCH ? AND rowid IN (SELECT value FROM json_each(?))
                ORDER BY bm25(chunks_fts),rowid LIMIT ?""", (expression, json.dumps(list(rows)), count))) if expression else []
        lexical_scores = dict(lexical)
        lexical_ranks = {row[0]: rank for rank, row in enumerate(lexical, 1)}
        vector_ranks = {chunk_id: rank for rank, chunk_id in enumerate(semantic, 1)}
        ranked, scores = rrf(list(lexical_scores), semantic, weights=RRF_WEIGHTS)
        # PDF/PNG/scan variants can contain identical passages; keep room for other evidence.
        seen, distinct = set(), []
        for chunk_id in ranked:
            if rows[chunk_id]["text"] not in seen:
                seen.add(rows[chunk_id]["text"])
                distinct.append(chunk_id)
        return [dict(chunk_id=i, document_id=rows[i]["document_id"], filename=rows[i]["filename"],
                     page_number=rows[i]["page_number"], text=rows[i]["text"], kind=rows[i]["kind"], rrf_score=scores[i],
                     lexical_rank=lexical_ranks.get(i), vector_rank=vector_ranks.get(i),
                     bm25=lexical_scores.get(i), cosine_similarity=cosine[i]) for i in distinct[:top_k]]

    def copy_documents(self, source_session, target_session, content_hashes):
        """Reuse indexed demo documents in a new scope without another embedding run."""
        with self.connection() as db:
            originals = db.execute(
                "SELECT * FROM documents WHERE session_id=? AND content_hash IN "
                "(SELECT value FROM json_each(?))",
                (source_session, json.dumps(content_hashes)),
            ).fetchall()
            for doc in originals:
                if db.execute("SELECT 1 FROM documents WHERE session_id=? AND content_hash=?",
                              (target_session, doc["content_hash"])).fetchone():
                    continue
                new_id = str(uuid4())
                db.execute("INSERT INTO documents(id,session_id,filename,content_hash,stored_path,pipeline_id) "
                           "VALUES(?,?,?,?,?,?)", (new_id, target_session, doc["filename"],
                           doc["content_hash"], doc["stored_path"], doc["pipeline_id"]))
                db.execute("INSERT INTO pages SELECT ?,page_number,raw_text,clean_text,method,blocks_json,warnings_json "
                           "FROM pages WHERE document_id=?", (new_id, doc["id"]))
                db.execute("INSERT INTO chunks(document_id,page_number,ordinal,text,kind,embedding) "
                           "SELECT ?,page_number,ordinal,text,kind,embedding FROM chunks WHERE document_id=?",
                           (new_id, doc["id"]))

    def documents(self, session_id=DOCUMENT_SESSION):
        with self.connection() as db:
            return [dict(row) for row in db.execute("""SELECT d.id,d.filename,
                (SELECT COUNT(*) FROM pages WHERE document_id=d.id) AS pages,
                (SELECT COUNT(*) FROM chunks WHERE document_id=d.id) AS chunks
                FROM documents d WHERE session_id=? ORDER BY d.created_at,d.id""", (session_id,))]

    def inspect(self, document_id, page_number=1, session_id=DOCUMENT_SESSION):
        with self.connection() as db:
            row = db.execute("""SELECT p.* FROM pages p JOIN documents d ON d.id=p.document_id
                WHERE d.session_id=? AND d.id=? AND p.page_number=?""", (session_id, document_id, page_number)).fetchone()
            if row is None:
                raise ValueError("Page not found in this session.")
            return dict(row)
