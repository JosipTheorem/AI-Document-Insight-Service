"""Evaluation must distinguish format, evidence and deliberate missing answers."""

import json
import unittest
from unittest.mock import Mock

from langchain_core.messages import AIMessage

from evaluation.runner import score_case, valid_format


class EvaluationTests(unittest.TestCase):
    def test_format_requires_raw_json_with_expected_fields_and_a_citation(self):
        answer = json.dumps({"article": "228 (1)", "answer": "Do tri godine.",
                             "source": "[kazneni_zakon_izvadak.pdf, str. 1]"})
        self.assertTrue(valid_format(answer))
        self.assertFalse(valid_format("```json\n" + answer + "\n```"))
        self.assertFalse(valid_format(answer.replace('"article"', '"unexpected_field"')))
        self.assertFalse(valid_format(answer.replace("str. 1", "str. 0")))

    def test_unanswerable_has_no_fabricated_exact_gold_hit(self):
        case = {"id": "missing", "question": "Missing fact?", "answerable": False,
                "gold_chunk": None, "expected_answer": {"article": "", "answer": "Not in excerpt.", "source": ""}}
        answer = json.dumps(case["expected_answer"])
        self.assertTrue(valid_format(answer))
        judge = Mock()
        judge.invoke.return_value = AIMessage('{"retrieval_ok":false,"answer_ok":true,"reason":"Missing fact."}')
        row = score_case(case, {"messages": [AIMessage(answer)]}, None, judge)
        self.assertIsNone(row["exact_gold_hit"])
        self.assertFalse(row["tool_called"])

    def test_invalid_judge_output_is_a_failure(self):
        judge = Mock()
        judge.invoke.return_value = AIMessage('{"retrieval_ok":"true"}')
        row = score_case({"id": "error", "question": "q", "answerable": False,
                          "gold_chunk": None, "expected_answer": {}}, None, "Agent unavailable", judge)
        self.assertEqual(row["error"], "Agent unavailable")
        self.assertIsNotNone(row["judge_error"])
        self.assertFalse(row["judge"]["answer_ok"])
