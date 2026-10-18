"""Offline checks: python -m unittest discover -s tests -v."""

import tempfile
import os
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pymupdf
from PIL import Image, ImageDraw
from langchain_core.utils.function_calling import convert_to_openai_tool

from config import DOCUMENT_SESSION
from retrieval import Index
from retrieval.documents import clean_text, extract, make_chunks, read_page
from retrieval.index import keywords, rrf
from tools.search import make_search_documents


def page(text, number=1, kind="text"):
    return dict(number=number, raw=text, blocks=[(kind, text)], method="native", warnings=[])


def fake_embeddings(texts, query=False):
    return np.array([[1, 0, 0] if "penalty" in t.lower() or "punishment" in t.lower()
                     else [0, 1, 0] for t in texts], dtype=np.float32)


class ProcessingTests(unittest.TestCase):
    def test_cleanup_keeps_amounts_ids_and_diacritics(self):
        text = "  Iznos:\t1.234,56 EUR\r\nNN-2024-57\x00\n\n\nČlanak 17."
        self.assertEqual(clean_text(text), "Iznos: 1.234,56 EUR\nNN-2024-57\n\nČlanak 17.")
        self.assertEqual(clean_text("transfor-\nmacije"), "transformacije")

    def test_chunks_keep_overlap_and_page_numbers(self):
        words = [f"word{i}" for i in range(25)]
        chunks = make_chunks([page(" ".join(words), 4), page("Next page", 5)], 10, 3)
        self.assertEqual(chunks[0]["text"].split()[-3:], chunks[1]["text"].split()[:3])
        self.assertTrue(all(len(c["text"].split()) <= 10 for c in chunks))
        self.assertEqual(chunks[-1]["page_number"], 5)
        self.assertTrue(set(words).issubset(set(" ".join(c["text"] for c in chunks).split())))

    def test_numbered_ocr_rules_keep_title_and_separate_facts(self):
        chunks = make_chunks([page("Traffic rules\n1. Speed limit.\n2. Alcohol limit.\n3. Seat belt.", kind="ocr")])
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all("Traffic rules" in c["text"] for c in chunks))
        self.assertIn("Speed", chunks[0]["text"])
        self.assertNotIn("Alcohol", chunks[0]["text"])

    def test_multi_page_pdf_keeps_text_and_page_attribution(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "two-pages.pdf"
            with pymupdf.open() as document:
                for marker in ("FIRST_PAGE", "SECOND_PAGE"):
                    document.new_page().insert_text((50, 70), marker + " contains a separate paragraph for this document.")
                document.save(path)
            with patch.dict(os.environ, {"APP_LOG_FILE": str(Path(temporary) / "events.txt")}):
                pages = extract(path)
        self.assertEqual([p["number"] for p in pages], [1, 2])
        self.assertIn("FIRST_PAGE", pages[0]["raw"])
        self.assertNotIn("SECOND_PAGE", pages[0]["raw"])
        self.assertIn("SECOND_PAGE", pages[1]["raw"])

    def test_columns_are_not_interleaved(self):
        with pymupdf.open() as document:
            pdf_page = document.new_page(width=600, height=500)
            pdf_page.insert_text((50, 40), "A document with two separate columns", fontsize=18)
            for start, end, box in [("LEFT_START", "LEFT_END", (50, 80, 285, 300)),
                                     ("RIGHT_START", "RIGHT_END", (315, 80, 550, 300))]:
                pdf_page.insert_textbox(box, start + "\n" + "This is a separate paragraph in one column. " * 6 + "\n" + end)
            text = read_page(pdf_page)["raw"]
        self.assertLess(text.index("LEFT_START"), text.index("LEFT_END"))
        self.assertLess(text.index("LEFT_END"), text.index("RIGHT_START"))
        self.assertLess(text.index("RIGHT_START"), text.index("RIGHT_END"))

    def test_legal_subsections_keep_article_and_do_not_mix_penalties(self):
        text = ("Članak 228. - Krađa\n(1) Osnovni oblik: do tri godine.\n"
                "(2) Mala vrijednost: do jedne godine.\n"
                "Članak 230. - Razbojništvo\n(1) Od jedne do deset godina.\n"
                "Izvori i važna napomena\nOvo je testni izvadak.")
        for kind in ("text", "ocr"):
            chunks = make_chunks([page(text, kind=kind)])
            theft = [c["text"] for c in chunks if c["text"].startswith("Članak 228.")]
            self.assertEqual(len(theft), 2)
            self.assertIn("(1) Osnovni oblik: do tri godine.", theft[0])
            self.assertNotIn("Mala vrijednost", theft[0])
            self.assertNotIn("Razbojništvo", " ".join(theft))
            self.assertFalse(any("Izvori" in c["text"] for c in chunks if c["text"].startswith("Članak")))

    def test_real_pdf_table_preserves_column_labels(self):
        with pymupdf.open() as document:
            pdf_page = document.new_page(width=600, height=400)
            rows = [["Product", "Quantity", "Amount"], ["Monitor", "2", "300.00 EUR"], ["Keyboard", "1", "50.00 EUR"]]
            xs, ys = [50, 240, 340, 520], [80, 125, 170, 215]
            for x in xs:
                pdf_page.draw_line((x, ys[0]), (x, ys[-1]))
            for y in ys:
                pdf_page.draw_line((xs[0], y), (xs[-1], y))
            for r, row in enumerate(rows):
                for c, text in enumerate(row):
                    pdf_page.insert_text((xs[c] + 8, ys[r] + 25), text)
            chunks = make_chunks([read_page(pdf_page)])
        self.assertTrue(any("Product: Monitor" in c["text"] and "Amount: 300.00 EUR" in c["text"] for c in chunks))

    @patch("retrieval.documents.rapidocr_api.exec_ocr",
           side_effect=lambda target, **kw: target.insert_text((60, 200), "Text inside an image"))
    def test_scanned_and_mixed_pdf_pages_use_ocr(self, ocr):
        image = BytesIO()
        picture = Image.new("RGB", (400, 200), "white")
        ImageDraw.Draw(picture).text((20, 40), "Text inside an image", fill="black", font_size=28)
        picture.save(image, format="PNG")
        with pymupdf.open() as document:
            pdf_page = document.new_page(width=600, height=600)
            pdf_page.insert_image(pymupdf.Rect(50, 150, 550, 450), stream=image.getvalue())
            self.assertEqual(read_page(pdf_page)["method"], "ocr")
            pdf_page.insert_text((50, 50), "This page has enough native text and an embedded picture.")
            extracted = read_page(pdf_page)
            self.assertEqual(extracted["method"], "native+ocr")
            self.assertIn("Text inside an image", extracted["raw"])

    def test_rrf_rewards_agreement(self):
        ranked, scores = rrf([10, 20], [30, 20])
        self.assertEqual(ranked, [20, 10, 30])
        self.assertAlmostEqual(scores[20], 2 / 62)
        self.assertAlmostEqual(scores[10], 1 / 61)
        _, weighted = rrf([10, 20], [30, 20], weights=(1, 2))
        self.assertAlmostEqual(weighted[20], 3 / 62)
        self.assertGreater(weighted[30], weighted[10])
        self.assertEqual(keywords('penalty: "ABC-123" OR (shoes*)'), '"penalty" OR "abc" OR "123" OR "shoes"')


class IndexTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.enterContext(patch.dict(os.environ, {"APP_LOG_FILE": str(self.root / "events.txt")}))
        self.index = Index(self.root / "index.sqlite3")
        self.index.model_id = "test-model@v1"
        self.index.embed = Mock(side_effect=fake_embeddings)
        extractor = patch("retrieval.index.extract", side_effect=lambda path: [
            page(text, i) for i, text in enumerate(path.read_text().split("\f"), 1)])
        extractor.start()
        self.addCleanup(extractor.stop)

    def ingest(self, name="penalty.pdf", text="Penalty punishment within 30 days.", session="alice", force=False):
        path = self.root / name
        path.write_text(text)
        return self.index.ingest(path, session, force)

    def test_deduplication_persistence_and_raw_text(self):
        document = self.ingest()
        self.assertEqual(self.ingest()["status"], "unchanged")
        self.assertEqual(self.index.embed.call_count, 1)
        restarted = Index(self.index.path)
        restarted.model_id, restarted.embed = self.index.model_id, self.index.embed
        self.assertIn("30 days", restarted.search("punishment", "alice")[0]["text"])
        self.assertEqual(restarted.inspect(document["document_id"], 1, "alice")["raw_text"], "Penalty punishment within 30 days.")

    def test_sessions_and_document_filters_apply_to_both_searches(self):
        self.ingest("secret.pdf", "Penalty punishment secret\f" * 40, "bob")
        own = self.ingest()
        hit = self.index.search("penalty punishment", "alice", top_k=1)[0]
        self.assertEqual(hit["document_id"], own["document_id"])
        self.assertEqual((hit["lexical_rank"], hit["vector_rank"]), (1, 1))
        self.assertNotIn("secret", hit["text"])
        self.assertEqual(self.index.search("penalty", "bob", document_ids=[own["document_id"]]), [])
        self.assertEqual(self.index.search("penalty", "alice", document_ids=[]), [])
        self.assertEqual(self.index.search("penalty", "nobody"), [])
        with self.assertRaises(ValueError):
            self.index.inspect(own["document_id"], 1, "bob")

    def test_semantic_search_without_keyword_matches(self):
        self.ingest(text="Penalty settlement within thirty days.")
        hit = self.index.search("punishment deadline", "alice", top_k=1)[0]
        self.assertIsNone(hit["lexical_rank"])
        self.assertEqual(hit["vector_rank"], 1)

    def test_identical_passages_do_not_crowd_out_other_evidence(self):
        first = self.ingest("native.pdf", "Penalty up to three years.")
        self.ingest("scan.pdf", "Penalty  up to three years.")
        other = self.ingest("other.pdf", "Different penalty: up to one year.")
        hits = self.index.search("penalty", "alice", top_k=3)
        self.assertEqual(len(hits), 2)
        self.assertEqual(len({hit["text"] for hit in hits}), 2)
        self.assertTrue({first["document_id"], other["document_id"]}.issubset({h["document_id"] for h in hits}))

    def test_reindex_is_atomic_and_fts_has_no_duplicates(self):
        self.ingest()
        self.ingest(force=True)
        self.index.embed.side_effect = RuntimeError("Embedding unavailable")
        with self.assertRaises(RuntimeError):
            self.ingest(force=True)
        with self.index.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'penalty'").fetchone()[0], 1)
            self.assertEqual(db.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_bad_inputs_and_model_mismatch_are_rejected(self):
        with self.assertRaises(ValueError):
            self.ingest(text=" \n\t")
        self.ingest()
        self.index.model_id = "different-model"
        with self.assertRaises(ValueError):
            self.index.search("penalty", "alice")
        self.index.model_id = "test-model@v1"
        self.index.embed.side_effect = lambda *a, **k: [[float("nan"), 0, 1]]
        with self.assertRaises(ValueError):
            self.index.search("penalty", "alice")
        for query in ["", "x" * 2001]:
            with self.assertRaises(ValueError):
                self.index.search(query, "alice")

    def test_fts_syntax_is_treated_as_text(self):
        self.ingest()
        for query in ['penalty: (OR *)', '"', '--', 'AND NOT']:
            self.index.search(query, "alice")


class ToolTests(unittest.TestCase):
    def test_tool_returns_sources_and_uses_the_configured_session(self):
        index = Mock()
        search_documents = make_search_documents(index=index)
        index.search.return_value = [dict(filename="penalty.pdf", page_number=2, text="Total: 30 EUR", embedding="private")]
        self.assertEqual(search_documents("total"), "[penalty.pdf, str. 2]\nTotal: 30 EUR")
        index.search.assert_called_once_with("total", session_id=DOCUMENT_SESSION, top_k=3)
        index.search.side_effect = ConnectionError("Ollama unavailable")
        self.assertIn("Document search failed", search_documents("total"))

    def test_model_sees_only_query_and_each_tool_keeps_its_server_session(self):
        index = Mock()
        index.search.return_value = []
        alice, bob = [make_search_documents(scope, index) for scope in ("alice", "bob")]
        schema = convert_to_openai_tool(alice)["function"]
        self.assertEqual(schema["name"], "search_documents")
        self.assertEqual(set(schema["parameters"]["properties"]), {"query"})
        self.assertIn("Pass the user's full question", schema["description"])
        self.assertNotIn("Bind the document scope", schema["description"])
        alice("one question")
        bob("another question")
        self.assertEqual([call.kwargs["session_id"] for call in index.search.call_args_list], ["alice", "bob"])


if __name__ == "__main__":
    unittest.main()
