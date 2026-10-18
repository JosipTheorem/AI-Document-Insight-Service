"""Embedding reuse, invalidation and fallback without a running Redis/Ollama."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from redis.exceptions import ConnectionError

from retrieval import Index
from retrieval.cache import EmbeddingCache


class CacheTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.log = root / "events.txt"
        self.enterContext(patch.dict(os.environ, {"APP_LOG_FILE": str(self.log)}))
        self.values = {}
        self.redis = MagicMock()
        self.redis.mget.side_effect = lambda keys: [self.values.get(key) for key in keys]
        self.pipe = self.redis.pipeline.return_value.__enter__.return_value
        self.pipe.set.side_effect = lambda key, value, **kw: self.values.update({key: value})
        self.enterContext(patch("retrieval.cache.Redis.from_url", return_value=self.redis))
        self.index = Index(root / "index.sqlite3")
        self.index.model_id = "model@digest-1"
        self.index.client = Mock()
        self.index.client.embed.side_effect = lambda **kw: Mock(
            embeddings=[[len(text), 1, 2] for text in kw["input"]])

    def test_repeated_and_partial_inputs_preserve_order_and_skip_inference(self):
        first = self.index.embed(["alpha", "beta", "alpha"])
        self.assertEqual(self.index.client.embed.call_args.kwargs["input"], ["alpha", "beta"])
        np.testing.assert_array_equal(first[0], first[2])
        second = self.index.embed(["beta", "new", "alpha"])
        self.assertEqual(self.index.client.embed.call_args.kwargs["input"], ["new"])
        np.testing.assert_array_equal(first[1], second[0])
        np.testing.assert_array_equal(first[0], second[2])
        self.index.client.embed.side_effect = AssertionError("Cache hit must skip Ollama")
        np.testing.assert_array_equal(second, self.index.embed(["beta", "new", "alpha"]))
        self.assertTrue(all(call.kwargs["ex"] == 604800 for call in self.pipe.set.call_args_list))

    def test_query_prefix_and_model_digest_separate_cache_entries(self):
        self.index.embed(["private question"])
        self.index.embed(["private question"], query=True)
        self.assertTrue(self.index.client.embed.call_args.kwargs["input"][0].endswith("Query: private question"))
        self.index.model_id = "model@digest-2"
        self.index.embed(["private question"], query=True)
        self.assertEqual(self.index.client.embed.call_count, 3)
        self.assertEqual(len(self.values), 3)
        self.assertTrue(all("private question" not in key for key in self.values))
        self.assertNotIn("private question", self.log.read_text())

    def test_cache_survives_a_new_index_instance(self):
        vector = self.index.embed(["shared passage"])
        restarted = Index(self.index.path)
        restarted.model_id = self.index.model_id
        restarted.client = Mock()
        restarted.client.embed.side_effect = AssertionError("Must reuse shared Redis cache")
        np.testing.assert_array_equal(vector, restarted.embed(["shared passage"]))

    def test_redis_read_failure_uses_ollama_and_waits_before_retry(self):
        self.redis.mget.side_effect = ConnectionError("Unavailable")
        self.assertEqual(self.index.embed(["first"]).shape, (1, 3))
        self.assertEqual(self.index.embed(["second"]).shape, (1, 3))
        self.assertEqual(self.redis.mget.call_count, 1)
        self.assertEqual(self.index.client.embed.call_count, 2)
        self.pipe.execute.assert_not_called()
        rows = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(sum(r["event"] == "embedding_cache_error" for r in rows), 1)
        self.index.cache.retry_after = 0
        self.redis.mget.side_effect = lambda keys: [self.values.get(key) for key in keys]
        self.index.embed(["first"])
        self.assertEqual(self.redis.mget.call_count, 2)

    def test_redis_write_failure_does_not_fail_the_embedding(self):
        self.pipe.execute.side_effect = ConnectionError("Unavailable")
        self.assertEqual(self.index.embed(["passage"]).shape, (1, 3))
        self.assertIn('"operation": "write"', self.log.read_text())

    def test_corrupt_entries_are_recomputed_and_invalid_model_output_is_not_cached(self):
        key = self.index.cache.key(self.index.model_id, "passage")
        for corrupt in [b"bad", b"", np.array([np.nan, 0, 1], dtype="<f4").tobytes(),
                        np.zeros(3, dtype="<f4").tobytes()]:
            self.values[key] = corrupt
            self.index.embed(["passage"])
            self.assertNotEqual(self.values[key], corrupt)
        self.index.client.embed.return_value = Mock(embeddings=[[0, 0, 0]])
        self.index.client.embed.side_effect = None
        with self.assertRaises(ValueError):
            self.index.embed(["new passage"])
        self.assertNotIn(self.index.cache.key(self.index.model_id, "new passage"), self.values)

    def test_disabled_cache_uses_ollama_without_redis(self):
        self.index.cache = EmbeddingCache(url="")
        self.index.embed(["passage"])
        self.index.embed(["passage"])
        self.assertEqual(self.index.client.embed.call_count, 2)
        self.redis.mget.assert_not_called()


if __name__ == "__main__":
    unittest.main()
