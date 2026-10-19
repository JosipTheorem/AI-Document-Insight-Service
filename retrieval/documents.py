"""Read PDF/images, clean text and make source-aware chunks."""

import json
import re
import unicodedata
from io import BytesIO
from pathlib import Path

import pymupdf
import pymupdf4llm
from pymupdf4llm.ocr import rapidocr_api
from PIL import Image, ImageOps

from config import CHUNK_WORDS, OVERLAP_WORDS
from app_logging import timed


def clean_text(text):
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    for char in ("\u00ad", "\u200b", "\ufeff"):
        text = text.replace(char, "")
    text = "".join(c for c in text if c in "\n\t" or unicodedata.category(c) != "Cc")
    text = re.sub(r"([^\W\d_]{2,})-\s*\n\s*([^\W\d_]{2,})", r"\1\2", text)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(" ".join(line.split()) for line in text.split("\n"))).strip()


def read_page(page):
    """Use PyMuPDF4LLM's layout parser and RapidOCR; retain page attribution."""
    native, used_ocr = bool(page.get_text().strip()), False

    def ocr(target, **kwargs):
        nonlocal used_ocr
        used_ocr = True
        return rapidocr_api.exec_ocr(target, **kwargs)

    result = json.loads(pymupdf4llm.to_json(page.parent, pages=[page.number],
                        ocr_function=ocr, ocr_dpi=200, show_progress=False))
    blocks = []
    for box in result["pages"][0]["boxes"]:
        if box["table"]:
            rows = box["table"]["extract"]
            # Keep every row: the first row is not necessarily a header.
            text = "\n".join(f"Row {i}: " + " | ".join(str(cell or "") for cell in row)
                             for i, row in enumerate(rows, 1))
            kind = "table"
        else:
            text = "\n".join("".join(span["text"] for span in line["spans"]) for line in box["textlines"] or [])
            kind = "ocr" if used_ocr else "text"
        if text.strip():
            blocks.append((kind, text))
    method = "native+ocr" if native and used_ocr else "ocr" if used_ocr else "native"
    return dict(number=page.number + 1, raw="\n\n".join(text for _, text in blocks), blocks=blocks,
                method=method, warnings=["OCR output may contain recognition errors."] if used_ocr else [])


@timed("extraction_completed")  #logging
def extract(path):
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        document = pymupdf.open(path)
    elif path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        with Image.open(path) as image:
            if image.format not in {"PNG", "JPEG"} or image.width * image.height > 40_000_000:
                raise ValueError("Expected a PNG/JPEG image of at most 40 megapixels.")
            image = ImageOps.exif_transpose(image).convert("RGB")
            buffer = BytesIO()
            image.save(buffer, format="PNG")
        with pymupdf.open(stream=buffer.getvalue(), filetype="png") as picture:
            document = pymupdf.open(stream=picture.convert_to_pdf(), filetype="pdf")
    else:
        raise ValueError("Supported files: PDF, PNG and JPEG.")
    with document:
        if document.needs_pass or not 1 <= len(document) <= 200:
            raise ValueError("PDF must be unlocked and contain 1 to 200 pages.")
        return [read_page(page) for page in document]


def make_chunks(pages, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    if not 0 <= overlap < size:
        raise ValueError("Chunk sizes must satisfy 0 <= overlap < size.")
    chunks = []
    for page in pages:
        sections, pending = [], []
        for kind, raw in page["blocks"]:
            lines = raw.splitlines()
            starts = [i for i, line in enumerate(lines) if re.match(r"^\s*\d{1,3}[.)]\s+", line)]
            if kind == "table" or (kind == "ocr" and len(starts) >= 3):
                if pending:
                    sections.append(("text", "\n\n".join(pending)))
                    pending = []
                if kind == "table":
                    sections.append(("table", raw))
                else:
                    heading = " ".join(" ".join(lines[:starts[0]]).split()[-40:])
                    sections += [("list", heading + "\n" + "\n".join(lines[a:b]))
                                 for a, b in zip(starts, starts[1:] + [len(lines)])]
            else:
                pending.append(raw)
        sections.append(("text", "\n\n".join(pending)))
        # Keep a legal article/subsection with its own heading, instead of mixing penalties.
        focused = []
        for kind, text in sections:
            parts = re.split(r"(?m)(?=^Članak \d+\.|^Izvori i važna napomena)", text)
            for part in parts:
                if re.match(r"Članak \d+\.", part) and "\n" in part.rstrip():
                    heading, _, body = part.partition("\n")
                    clauses = re.split(r"(?m)(?=^\(\d+\)\s)", body)
                    focused.extend((kind, heading + "\n" + clause) for clause in clauses if clause.strip())
                else:
                    focused.append((kind, part))
        for kind, text in focused:
            words = clean_text(text).split()
            for start in range(0, len(words), size - overlap):
                chunks.append(dict(page_number=page["number"], text=" ".join(words[start:start + size]), kind=kind))
                if start + size >= len(words):
                    break
    return chunks
