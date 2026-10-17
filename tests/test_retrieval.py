"""Offline checks: python -m unittest discover -s tests -v."""

import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pymupdf
from PIL import Image

from config.retrieval import DOCUMENT_SESSION
from retrieval import Index
from retrieval.documents import clean_text, column_order, make_chunks, read_page
from retrieval.index import keywords, rrf
from tools.search_documents.tool import search_documents


def page(text, number=1, kind="text"):
    return dict(number=number, raw=text, blocks=[(kind, text)], method="native", warnings=[])


def fake_embeddings(texts, query=False):
    return np.array([[1, 0, 0] if "invoice" in t.lower() or "payment" in t.lower()
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
        chunks = make_chunks([page("Etiquette\n1. Phone away.\n2. Listen first.\n3. Respect others.", kind="ocr")])
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all("Etiquette" in c["text"] for c in chunks))
        self.assertIn("Phone", chunks[0]["text"])
        self.assertNotIn("Listen", chunks[0]["text"])

    def test_columns_are_not_interleaved(self):
        blocks = [(pymupdf.Rect(box), ("text", text)) for box, text in [
            ((310, 80, 550, 130), "right1"), ((50, 140, 290, 180), "left2"),
            ((50, 20, 550, 50), "heading"), ((50, 80, 290, 130), "left1"),
            ((310, 140, 550, 180), "right2")]]
        self.assertEqual([text for _, text in column_order(blocks, 600)],
                         ["heading", "left1", "left2", "right1", "right2"])

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

    @patch("retrieval.documents.read_image", return_value=("Text inside an image", []))
    def test_scanned_and_mixed_pdf_pages_use_ocr(self, ocr):
        image = BytesIO()
        Image.new("RGB", (400, 200), "white").save(image, format="PNG")
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
        self.assertEqual(keywords('invoice: "ABC-123" OR (shoes*)'), '"invoice" OR "abc" OR "123" OR "shoes"')


class IndexTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = Index(self.root / "index.sqlite3")
        self.index.model_id = "test-model@v1"
        self.index.embed = Mock(side_effect=fake_embeddings)
        extractor = patch("retrieval.index.extract", side_effect=lambda path: [
            page(text, i) for i, text in enumerate(path.read_text().split("\f"), 1)])
        extractor.start()
        self.addCleanup(extractor.stop)

    def ingest(self, name="invoice.pdf", text="Invoice payment within 30 days.", session="alice", force=False):
        path = self.root / name
        path.write_text(text)
        return self.index.ingest(path, session, force)

    def test_deduplication_persistence_and_raw_text(self):
        document = self.ingest()
        self.assertEqual(self.ingest()["status"], "unchanged")
        self.assertEqual(self.index.embed.call_count, 1)
        restarted = Index(self.index.path)
        restarted.model_id, restarted.embed = self.index.model_id, self.index.embed
        self.assertIn("30 days", restarted.search("payment", "alice")[0]["text"])
        self.assertEqual(restarted.inspect(document["document_id"], 1, "alice")["raw_text"], "Invoice payment within 30 days.")

    def test_sessions_and_document_filters_apply_to_both_searches(self):
        self.ingest("secret.pdf", "Invoice payment secret\f" * 40, "bob")
        own = self.ingest()
        hit = self.index.search("invoice payment", "alice", top_k=1)[0]
        self.assertEqual(hit["document_id"], own["document_id"])
        self.assertEqual((hit["lexical_rank"], hit["vector_rank"]), (1, 1))
        self.assertNotIn("secret", hit["text"])
        self.assertEqual(self.index.search("invoice", "bob", document_ids=[own["document_id"]]), [])
        self.assertEqual(self.index.search("invoice", "alice", document_ids=[]), [])
        self.assertEqual(self.index.search("invoice", "nobody"), [])
        with self.assertRaises(ValueError):
            self.index.inspect(own["document_id"], 1, "bob")

    def test_semantic_search_without_keyword_matches(self):
        self.ingest(text="Invoice settlement within thirty days.")
        hit = self.index.search("payment deadline", "alice", top_k=1)[0]
        self.assertIsNone(hit["lexical_rank"])
        self.assertEqual(hit["vector_rank"], 1)

    def test_reindex_is_atomic_and_fts_has_no_duplicates(self):
        self.ingest()
        self.ingest(force=True)
        self.index.embed.side_effect = RuntimeError("Embedding unavailable")
        with self.assertRaises(RuntimeError):
            self.ingest(force=True)
        with self.index.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'invoice'").fetchone()[0], 1)
            self.assertEqual(db.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_bad_inputs_and_model_mismatch_are_rejected(self):
        with self.assertRaises(ValueError):
            self.ingest(text=" \n\t")
        self.ingest()
        self.index.model_id = "different-model"
        with self.assertRaises(ValueError):
            self.index.search("invoice", "alice")
        self.index.model_id = "test-model@v1"
        self.index.embed.side_effect = lambda *a, **k: [[float("nan"), 0, 1]]
        with self.assertRaises(ValueError):
            self.index.search("invoice", "alice")
        for query in ["", "x" * 2001]:
            with self.assertRaises(ValueError):
                self.index.search(query, "alice")

    def test_fts_syntax_is_treated_as_text(self):
        self.ingest()
        for query in ['invoice: (OR *)', '"', '--', 'AND NOT']:
            self.index.search(query, "alice")


class ToolTests(unittest.TestCase):
    @patch("tools.search_documents.tool._index")
    def test_tool_returns_sources_and_uses_the_configured_session(self, get_index):
        get_index().search.return_value = [dict(filename="invoice.pdf", page_number=2, text="Total: 30 EUR", embedding="private")]
        self.assertEqual(search_documents("total"), "[invoice.pdf, str. 2]\nTotal: 30 EUR")
        get_index().search.assert_called_once_with("total", session_id=DOCUMENT_SESSION, top_k=3)
        get_index().search.side_effect = ConnectionError("Ollama unavailable")
        self.assertIn("Document search failed", search_documents("total"))


if __name__ == "__main__":
    unittest.main()
