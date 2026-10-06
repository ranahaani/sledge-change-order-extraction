"""PDF text extraction with an OCR fallback for sparse pages.

Text-based PDFs use pypdf. A page with almost no text is treated as a scan:
embedded images are read with Pillow and passed to Tesseract when the
binary is installed. Without Tesseract the page stays empty and the
pipeline returns missing fields instead of guessing.
"""

from __future__ import annotations

import io
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader

_MIN_ALNUM = 40


@dataclass
class LoadedDocument:
    text: str
    ocr_used: bool = False
    page_count: int = 0
    warnings: list[str] = field(default_factory=list)


def load_path(path: Path) -> LoadedDocument:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return LoadedDocument(text=path.read_text(encoding="utf-8"), page_count=1)
    if suffix == ".pdf":
        return read_pdf(path)
    raise ValueError(f"unsupported file type: {suffix or 'unknown'}")


def load_bytes(data: bytes, suffix: str) -> LoadedDocument:
    if suffix in {".txt", ".md"}:
        return LoadedDocument(text=data.decode("utf-8"), page_count=1)
    if suffix == ".pdf":
        return read_pdf(io.BytesIO(data))
    raise ValueError(f"unsupported file type: {suffix or 'unknown'}")


def read_pdf(source: str | Path | BinaryIO) -> LoadedDocument:
    reader = PdfReader(source)
    pages: list[str] = []
    ocr_used = False
    warnings: list[str] = []
    tesseract = shutil.which("tesseract")
    for page in reader.pages:
        text = page.extract_text() or ""
        if _alnum_count(text) < _MIN_ALNUM:
            if tesseract is None:
                warnings.append("ocr_unavailable: tesseract is not on PATH")
            else:
                ocr_text = _ocr_page(page)
                if _alnum_count(ocr_text) > _alnum_count(text):
                    text = ocr_text
                    ocr_used = True
        pages.append(text)
    if not any(part.strip() for part in pages) and not ocr_used:
        warnings.append("no_text_extracted")
    return LoadedDocument(
        text="\n\n".join(pages),
        ocr_used=ocr_used,
        page_count=len(reader.pages),
        warnings=_unique(warnings),
    )


def _ocr_page(page: object) -> str:
    import pytesseract
    from PIL import Image

    chunks: list[str] = []
    images = getattr(page, "images", [])
    for image in images:
        data = getattr(image, "data", None)
        if not data:
            continue
        try:
            with Image.open(io.BytesIO(data)) as img:
                chunks.append(pytesseract.image_to_string(img))
        except (OSError, pytesseract.TesseractError):
            continue
    return "\n".join(chunks)


def _alnum_count(text: str) -> int:
    return sum(character.isalnum() for character in text)


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered
