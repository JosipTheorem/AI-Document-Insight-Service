"""Pure functions so log parsing, percentiles and token buckets are testable."""

import json

import pandas as pd

COLUMNS = ["timestamp", "event_id", "event", "level", "request_id", "run_id", "interface",
           "path", "status_code", "duration_ms", "input_tokens", "output_tokens",
           "usage_available", "generation_ms", "purpose", "model"]


def read_events(path):
    records, skipped = [], 0
    if path.exists():
        with path.open(encoding="utf-8", errors="replace") as source:
            for line in source:
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or not row.get("timestamp") or not row.get("event"):
                        raise ValueError("Not an event")
                    records.append(row)
                except (ValueError, TypeError):
                    skipped += 1
    frame = pd.DataFrame(records)
    for column in COLUMNS:
        if column not in frame:
            frame[column] = pd.Series(index=frame.index, dtype="object")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601", errors="coerce")
    skipped += int(frame["timestamp"].isna().sum())
    frame = frame.dropna(subset=["timestamp"])
    frame["timestamp"] = frame["timestamp"].dt.tz_convert("Europe/Zagreb")
    for column in ("status_code", "duration_ms", "input_tokens", "output_tokens", "generation_ms"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("timestamp"), skipped


def filter_days(events, first, last):
    days = events["timestamp"].dt.date
    return events[(days >= first) & (days <= last)].copy()


def summarize(events):
    http = events[events["event"] == "http_request"]
    latency = events[((events["event"] == "http_request") & (events["path"] == "/ask")
                      | (events["event"] == "terminal_request")) & (events["status_code"] == 200)]["duration_ms"] / 1000
    calls = events[events["event"] == "llm_completed"]
    measured = calls[calls["usage_available"] == True]
    failures = events[events["level"] == "ERROR"]
    error_ids = failures["request_id"].fillna(failures["run_id"]).fillna(failures["event_id"])
    return {
        "requests": len(http), "answers": len(latency),
        "median": latency.median(), "p95": latency.quantile(.95), "p99": latency.quantile(.99),
        "input_tokens": int(measured["input_tokens"].sum()),
        "output_tokens": int(measured["output_tokens"].sum()),
        "errors": error_ids.nunique(),
        "rejected": int(http["status_code"].between(400, 499).sum()),
        "blocked": int((events["event"] == "security_blocked").sum()),
        "missing_usage": int(len(calls) - len(measured)),
    }


def token_buckets(events, frequency):
    calls = events[(events["event"] == "llm_completed") & (events["usage_available"] == True)].copy()
    if calls.empty:
        return pd.DataFrame(columns=["input_tokens", "output_tokens"])
    # Second-level charts show the last 15 minutes with data; avoid millions of empty bins.
    if frequency == "s":
        calls = calls[calls["timestamp"] >= calls["timestamp"].max() - pd.Timedelta(minutes=15)]
    return calls.set_index("timestamp")[["input_tokens", "output_tokens"]].resample(frequency).sum()
