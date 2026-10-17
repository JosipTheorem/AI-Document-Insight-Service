"""Read PDF/images, clean text and make source-aware chunks."""

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageOps

from config.retrieval import CHUNK_WORDS, OVERLAP_WORDS
from app_logging import timed

pymupdf.no_recommend_layout()


def clean_text(text):
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    for char in ("\u00ad", "\u200b", "\ufeff"):
        text = text.replace(char, "")
    text = "".join(c for c in text if c in "\n\t" or unicodedata.category(c) != "Cc")
    text = re.sub(r"([^\W\d_]{2,})-\s*\n\s*([^\W\d_]{2,})", r"\1\2", text)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(" ".join(line.split()) for line in text.split("\n"))).strip()


@lru_cache(maxsize=1)
def _ocr_engine():
    from rapidocr import RapidOCR
    return RapidOCR(params={"Global.log_level": "warning"})


def read_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = min(3, max(1, 1000 / image.width))
    image = image.resize((round(image.width * scale), round(image.height * scale)))
    result = _ocr_engine()(np.asarray(image))
    warnings = ["OCR contains uncertain text."] if any(s < 0.8 for s in (result.scores or [])) else []
    return "\n".join(result.txts or []), warnings


def column_order(blocks, width):
    """Read each column top to bottom, keeping full-width headings in place."""
    middle = width / 2
    anchors = [b for b in blocks if b[0].x0 < middle - 8 and b[0].x1 > middle + 8]
    remaining = [b for b in blocks if b not in anchors]
    ordered = []
    for anchor in sorted(anchors, key=lambda b: b[0][1]) + [None]:
        band = [b for b in remaining if anchor is None or b[0][1] < anchor[0][1]]
        left = [b for b in band if b[0][2] <= middle + 8]
        right = [b for b in band if b[0][0] >= middle - 8 and b not in left]
        if len(left) >= 2 and len(right) >= 2 and len(left) + len(right) == len(band):
            ordered += sorted(left, key=lambda b: b[0][1]) + sorted(right, key=lambda b: b[0][1])
        else:
            ordered += sorted(band, key=lambda b: (b[0][1], b[0][0]))
        remaining = [b for b in remaining if b not in band]
        if anchor:
            ordered.append(anchor)
    return [block for _, block in ordered]


def read_page(page):
    raw, method, warnings = page.get_text(), "native", []
    if len("".join(raw.split())) < 30:
        pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csRGB, alpha=False)
        raw, warnings = read_image(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        return dict(number=page.number + 1, raw=raw, blocks=[("ocr", raw)], method="ocr", warnings=warnings)
    tables, blocks = [], []
    try:
        for table in page.find_tables().tables:
            rows = [[" ".join(str(cell or "").split()) for cell in row] for row in table.extract()]
            if len(rows) >= 2 and len(rows[0]) >= 2:
                header = rows[0]
                text = "\n".join("; ".join(f"{h or 'Column'}: {v}" for h, v in zip(header, row)) for row in rows[1:])
                tables.append((pymupdf.Rect(table.bbox), ("table", text)))
    except (RuntimeError, ValueError):
        warnings.append("Table detection failed; native text retained.")
    for item in page.get_text("blocks"):
        rect = pymupdf.Rect(item[:4])
        if item[6] == 0 and not any((rect & box).get_area() >= rect.get_area() * 0.95 for box, _ in tables):
            blocks.append((rect, ("text", item[4])))
    blocks += tables
    seen = set()
    for info in page.get_image_info():
        rect = pymupdf.Rect(info["bbox"]) & page.rect
        key = tuple(round(v) for v in rect)
        if key in seen or rect.get_area() < page.rect.get_area() * 0.08 or len(page.get_textbox(rect).strip()) >= 100:
            continue
        seen.add(key)
        pix = page.get_pixmap(clip=rect, dpi=200, colorspace=pymupdf.csRGB, alpha=False)
        text, notes = read_image(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        warnings += notes
        if text:
            blocks.append((rect, ("ocr", text)))
            raw += "\n\n" + text
            method = "native+ocr"
    return dict(number=page.number + 1, raw=raw, blocks=column_order(blocks, page.rect.width), method=method, warnings=warnings)


@timed("extraction_completed")  #logging
def extract(path):
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        with pymupdf.open(path) as document:
            if document.needs_pass or len(document) > 200:
                raise ValueError("PDF must be unlocked and contain at most 200 pages.")
            return [read_page(page) for page in document]
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Supported files: PDF, PNG and JPEG.")
    with Image.open(path) as image:
        if image.format not in {"PNG", "JPEG"} or image.width * image.height > 40_000_000:
            raise ValueError("Expected a PNG/JPEG image of at most 40 megapixels.")
        raw, warnings = read_image(image)
    return [dict(number=1, raw=raw, blocks=[("ocr", raw)], method="ocr", warnings=warnings)]


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
                    sections += [("table", line) for line in lines if line.strip()]
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
                if re.match(r"Članak \d+\.", part):
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
