"""Run real API uploads, 13 QA cases and a few integration checks against Ollama.

python -m evaluation.simulations
Models and retrieval are real; the observer only saves graph messages for judging.
All simulation data/logs use a separate local directory, not users' sessions.
"""

import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

from agent import build_agent
from api import create_app
from before_model.security import guard
from config.settings import PROJECT_ROOT
from evaluation.runner import FORMAT_REQUEST, make_judge, new_report, score_case
from retrieval import Index


class ObservedAgent:
    def __init__(self, scope, index):
        self.agent, self.result = build_agent(scope, index), None

    def invoke(self, *args, **kwargs):
        self.result = self.agent.invoke(*args, **kwargs)
        return self.result


def main():
    root = PROJECT_ROOT
    output = root / "evaluation/results"
    output.mkdir(exist_ok=True)
    os.environ["APP_LOG_FILE"] = str(root / "data/simulation-events.txt")
    dataset = json.loads((root / "evaluation/dataset/all_questions.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "files/sources.json").read_text(encoding="utf-8"))
    report, observed, checks = new_report(), {}, []
    report["checks"] = checks

    def save():
        (output / "simulations.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def check(name, condition, **details):
        checks.append({"name": name, "passed": bool(condition), **details})
        print(name, "PASS" if condition else "FAIL", flush=True)
        save()

    def observe(scope, index):
        observed[scope] = ObservedAgent(scope, index)
        return observed[scope]

    with TemporaryDirectory(prefix="abysalto-legal-") as temporary:
        index = Index(Path(temporary) / "index.sqlite3")
        with patch("api.build_agent", side_effect=observe), TestClient(create_app(index)) as client:
            def session(examples=False):
                response = client.post("/sessions", json={"include_examples": examples})
                response.raise_for_status()
                return response.json()["session_id"]

            scopes = {}
            # Each format is isolated so PDF search cannot hide an OCR failure.
            for kind in ("native", "image", "scan"):
                scope = scopes[kind] = session()
                for item in manifest:
                    relative = item["file"]
                    if not relative.startswith(("pdf/", "png/")):
                        continue
                    item_kind = "image" if relative.startswith("png/") else "scan" if relative.endswith("_scan.pdf") else "native"
                    if item_kind != kind:
                        continue
                    path = root / "files" / relative
                    response = client.post("/upload", data={"session_id": scope},
                                           files={"file": (path.name, path.read_bytes())})
                    check(f"upload:{path.name}", response.status_code == 200,
                          status=response.status_code, response=response.json())
                    if response.status_code != 200:
                        continue
                    page = index.inspect(response.json()["document_id"], session_id=scope)
                    check(f"extraction:{path.name}", page["method"] == ("native" if kind == "native" else "ocr"),
                          method=page["method"], warnings=json.loads(page["warnings_json"]))
                    index.copy_documents(scope, "demo", [item["sha256"]])
                question = dataset["cases"][0]
                started = time.perf_counter()
                response = client.post("/ask", json={"session_id": scope, "message": question["question"]})
                expected_fact = question["expected_answer"]["answer"].rstrip(".").casefold()
                check(f"chat:{kind}", response.status_code == 200 and bool(response.json().get("sources"))
                      and expected_fact in response.json().get("answer", "").casefold(),
                      answer=response.json().get("answer"), sources=response.json().get("sources"),
                      seconds=round(time.perf_counter() - started, 2))

            # Check all 13 supplied references via actual HTTP + agent + tools + 9B judge.
            judge = make_judge()
            for case in dataset["cases"]:
                scope = session(examples=True)
                started = time.perf_counter()
                print(f"API QA {case['id']}: {case['question']}", flush=True)
                response = client.post("/ask", json={"session_id": scope, "message": case["question"] + FORMAT_REQUEST})
                agent_seconds = round(time.perf_counter() - started, 2)
                row = score_case(case, observed[scope].result,
                                 None if response.status_code == 200 else response.text, judge)
                row.update(http_status=response.status_code, agent_seconds=agent_seconds,
                           seconds=round(time.perf_counter() - started, 2))
                report["cases"].append(row)
                save()
                print(f"  search={row['tool_called']}, JSON={row['strict_json']}, correct={row['judge']['answer_ok']}", flush=True)

            empty = session()
            result = client.post("/ask", json={"session_id": empty, "message": dataset["cases"][0]["question"]})
            check("empty_session_has_no_sources", result.status_code == 200 and result.json()["sources"] == [],
                  answer=result.json().get("answer"))
            # Exactly ten requests, even rejected ones; the eleventh must not run a model.
            limited = session()
            for _ in range(10):
                last = client.post("/ask", json={"session_id": limited, "message": "Jebi se."})
            calls_before = observed[limited].result
            last = client.post("/ask", json={"session_id": limited, "message": "Bok"})
            check("eleventh_message_blocked", last.status_code == 429 and observed[limited].result is calls_before,
                  status=last.status_code)

            path = root / "files/pdf/kazneni_zakon_izvadak.pdf"
            duplicate = client.post("/upload", data={"session_id": scopes["native"]},
                                    files={"file": (path.name, path.read_bytes())})
            check("duplicate_not_reembedded", duplicate.status_code == 200 and duplicate.json()["status"] == "unchanged")
            invalid = client.post("/upload", data={"session_id": empty}, files={"file": ("fake.pdf", b"not a PDF")})
            check("invalid_upload_rejected", invalid.status_code == 400)
            check("empty_session_stays_empty", not client.get("/documents", params={"session_id": empty}).json()["documents"])

    probes = json.loads((root / "tests/security_examples.json").read_text(encoding="utf-8"))
    probes += [{"message": case["question"] + FORMAT_REQUEST, "decision": "OK"} for case in dataset["cases"]]
    security = []
    for probe in probes:
        try:
            actual = guard.invoke([("system", (root / "before_model/prompt.txt").read_text(encoding="utf-8")),
                                   ("human", probe["message"])]).decision
            error = None
        except Exception as failure:
            actual, error = None, str(failure)
        security.append({**probe, "actual": actual, "error": error, "passed": actual == probe["decision"]})
        print("Security", len(security), security[-1]["passed"], flush=True)
    (output / "security.json").write_text(json.dumps(security, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Checks: {sum(c['passed'] for c in checks)}/{len(checks)}; "
          f"QA: {sum(c['judge']['answer_ok'] for c in report['cases'])}/{len(report['cases'])}; "
          f"security: {sum(c['passed'] for c in security)}/{len(security)}", flush=True)


if __name__ == "__main__":
    main()
