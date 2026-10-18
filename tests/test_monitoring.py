"""Known log fixtures check token accounting, time zones, percentiles and errors."""

import json
import os
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app_logging import bind_session, log_context, log_event
from app_logging.callbacks import UsageLogger
from monitoring.data import filter_days, read_events, summarize, token_buckets


class MonitoringTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(TemporaryDirectory())
        self.path = Path(directory) / "application.txt"
        self.enterContext(patch.dict(os.environ, {"APP_LOG_FILE": str(self.path)}))

    def test_callbacks_record_provider_counts_and_omit_message_contents(self):
        callback = UsageLogger("security", "local-model")
        run_id = uuid4()
        with log_context("api", request_id="request-1"):
            bind_session("SECRET_SESSION")
            callback.on_chat_model_start({}, [[HumanMessage("SECRET_INPUT")]], run_id=run_id)
            message = AIMessage("SECRET_OUTPUT", usage_metadata={"input_tokens": 13, "output_tokens": 7, "total_tokens": 20},
                                response_metadata={"eval_duration": 1_000_000_000})
            callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run_id)
        raw = self.path.read_text(encoding="utf-8")
        self.assertNotIn("SECRET", raw)
        rows, skipped = read_events(self.path)
        self.assertEqual(skipped, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.iloc[0]["request_id"], "request-1")
        self.assertEqual(rows.iloc[0]["generation_ms"], 1000)
        self.assertEqual(summarize(rows)["input_tokens"], 13)
        self.assertEqual(summarize(rows)["output_tokens"], 7)

    def test_missing_usage_is_not_estimated_and_model_errors_are_logged(self):
        callback = UsageLogger("agent", "local-model")
        run_id = uuid4()
        callback.on_chat_model_start({}, [], run_id=run_id)
        callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage("lots of words"))]]), run_id=run_id)
        error_id = uuid4()
        callback.on_chat_model_start({}, [], run_id=error_id)
        callback.on_llm_error(ConnectionError("SECRET_ERROR_DETAILS"), run_id=error_id)
        rows, _ = read_events(self.path)
        summary = summarize(rows)
        self.assertEqual(summary["missing_usage"], 1)
        self.assertEqual(summary["input_tokens"], 0)
        self.assertEqual(summary["output_tokens"], 0)
        self.assertEqual(summary["errors"], 1)
        self.assertNotIn("SECRET_ERROR_DETAILS", self.path.read_text())

    def test_percentiles_error_deduplication_and_completed_call_buckets(self):
        rows = []
        for i, milliseconds in enumerate([1000, 2000, 3000, 4000, 5000]):
            rows.append({"event": "http_request", "event_id": str(i), "request_id": str(i),
                         "timestamp": f"2026-10-17T10:00:0{i}+00:00", "path": "/ask", "status_code": 200,
                         "duration_ms": milliseconds, "level": "INFO"})
        rows += [
            {"event": "llm_completed", "timestamp": "2026-10-17T10:00:01Z", "usage_available": True, "input_tokens": 10, "output_tokens": 2},
            {"event": "llm_completed", "timestamp": "2026-10-17T10:00:03Z", "usage_available": True, "input_tokens": 20, "output_tokens": 3},
            {"event": "agent_error", "timestamp": "2026-10-17T10:01:00Z", "level": "ERROR", "request_id": "failed"},
            {"event": "http_request", "timestamp": "2026-10-17T10:01:00Z", "level": "ERROR", "request_id": "failed", "status_code": 503},
            {"event": "http_request", "timestamp": "2026-10-17T10:02:00Z", "level": "WARNING", "request_id": "limited", "status_code": 429},
        ]
        self.path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        events, _ = read_events(self.path)
        result = summarize(events)
        self.assertEqual(result["median"], 3)
        self.assertAlmostEqual(result["p95"], 4.8)
        self.assertAlmostEqual(result["p99"], 4.96)
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["rejected"], 1)
        self.assertEqual(token_buckets(events, "min")["input_tokens"].sum(), 30)
        seconds = token_buckets(events, "s")
        self.assertEqual(seconds["output_tokens"].sum(), 5)
        self.assertEqual(seconds.iloc[1]["output_tokens"], 0)

    def test_local_day_filter_and_malformed_tail(self):
        rows = [
            {"event": "http_request", "timestamp": "2026-10-16T22:30:00Z"},
            {"event": "http_request", "timestamp": "2026-10-17T22:30:00Z"},
            {"event": "invalid", "timestamp": "invalid-date"},
        ]
        self.path.write_text("\n".join(json.dumps(row) for row in rows) + '\n{"unfinished":', encoding="utf-8")
        events, skipped = read_events(self.path)
        self.assertEqual(skipped, 2)
        self.assertEqual(len(filter_days(events, date(2026, 10, 17), date(2026, 10, 17))), 1)
        missing, _ = read_events(self.path.parent / "missing.txt")
        self.assertEqual(summarize(missing)["answers"], 0)

    def test_dashboard_renders_real_log_file_and_empty_date_filter(self):
        from streamlit.testing.v1 import AppTest

        log_event("http_request", path="/ask", status_code=200, duration_ms=2000)  #logging
        log_event("llm_completed", model="test", purpose="agent", usage_available=True,  #logging
                  input_tokens=12, output_tokens=4, duration_ms=1000)
        log_event("embedding_cache", hits=3, misses=1, purpose="query")  #logging
        log_event("embedding_cache_error", level="WARNING", operation="read", error_type="ConnectionError")  #logging
        events, _ = read_events(self.path)
        summary = summarize(events)
        self.assertEqual((summary["cache_hits"], summary["cache_misses"], summary["cache_errors"]), (3, 1, 1))
        dashboard = Path(__file__).resolve().parents[1] / "monitoring/dashboard.py"
        ui = AppTest.from_file(str(dashboard)).run(timeout=20)
        self.assertFalse(ui.exception)
        self.assertTrue(any(metric.label == "Ulazni tokeni" and metric.value == "12" for metric in ui.metric))
        self.assertTrue(any(metric.label == "Udio pogodaka" and metric.value == "75%" for metric in ui.metric))
        ui.sidebar.date_input[0].set_value((date(2020, 1, 1), date(2020, 1, 1))).run(timeout=20)
        self.assertFalse(ui.exception)
        self.assertTrue(ui.info)


if __name__ == "__main__":
    unittest.main()
