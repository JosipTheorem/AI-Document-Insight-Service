"""Optional Redis embedding cache. SQLite remains the document index."""

from hashlib import sha256
from time import monotonic

import numpy as np
from redis import Redis, RedisError
from redis.backoff import NoBackoff
from redis.retry import Retry

from app_logging import log_event
from config import EMBEDDING_CACHE_TTL, REDIS_URL


class EmbeddingCache:
    def __init__(self, url=REDIS_URL, ttl=EMBEDDING_CACHE_TTL):
        if ttl <= 0:
            raise ValueError("Embedding cache TTL must be positive.")
        self.ttl, self.retry_after = ttl, 0
        self.client = Redis.from_url(url, socket_connect_timeout=0.25, socket_timeout=0.25,
                                     retry=Retry(NoBackoff(), 0)) if url else None

    @staticmethod
    def key(model_id, text, query=False):
        digest = sha256(f"{model_id}\0{text}".encode("utf-8")).hexdigest()
        return f"embedding:v1:{'query' if query else 'document'}:{digest}"

    def available(self):
        return self.client is not None and monotonic() >= self.retry_after

    def failed(self, operation, error):
        self.retry_after = monotonic() + 30
        log_event("embedding_cache_error", level="WARNING", operation=operation,  #logging
                  error_type=type(error).__name__)

    def get_many(self, keys):
        if not keys or not self.available():
            return {}
        try:
            values = self.client.mget(keys)
        except RedisError as error:
            self.failed("read", error)
            return {}
        cached, invalid = {}, 0
        for key, value in zip(keys, values, strict=True):
            if value is None:
                continue
            try:
                vector = np.frombuffer(value, dtype="<f4")
                if not vector.size or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1):
                    raise ValueError("Invalid cached vector.")
                cached[key] = vector
            except ValueError:
                invalid += 1  # Treat malformed cache entries as misses and overwrite them.
        log_event("embedding_cache", hits=len(cached), misses=len(keys) - len(cached),  #logging
                  invalid=invalid, purpose=keys[0].split(":")[2])
        return cached

    def put_many(self, vectors):
        if not vectors or not self.available():
            return
        try:
            with self.client.pipeline(transaction=False) as pipe:
                for key, vector in vectors.items():
                    pipe.set(key, np.asarray(vector, dtype="<f4").tobytes(), ex=self.ttl)
                pipe.execute()
        except RedisError as error:
            self.failed("write", error)
