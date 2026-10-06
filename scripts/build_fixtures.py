"""Rebuild the PDF fixtures committed under samples/pdf.

The text PDF is a real text layer (no OCR). The scan is an image-only PDF
with no text layer, so the pipeline has to OCR it. Re-run from the repo root:

    python scripts/build_fixtures.py
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "samples" / "pdf"
TEXT_SOURCE = ROOT / "samples" / "co_01_aia_clean.txt"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

SCAN_LINES = [
    "CHANGE ORDER",
    "",
    "Project: Cedar Clinic TI",
    "Project No: CC-310",
    "CO No: 6",
    "Date: August 8, 2026",
    "Owner: Cedar Clinic Partners",
    "Contractor: Holm Electric",
    "",
    "Description: Relocate three exit signs per the revised life safety plan.",
    "",
    "Description          Qty   Unit   Unit Price    Amount",
    "Exit sign relocation   3   EA     185.00        555.00",
    "After-hours labor      4   HR     110.00        440.00",
    "",
    "Subtotal  995.00",
    "Markup  0.00",
    "Tax  0.00",
    "Total  995.00",
    "",
    "Schedule impact: 1 day",
    "Reason: Code revision",
    "Approval: Pending",
]


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    text = TEXT_SOURCE.read_text(encoding="utf-8").splitlines()
    (PDF_DIR / "co_01_aia_clean.pdf").write_bytes(build_text_pdf(text, lines_per_page=16))
    render_scan(PDF_DIR / "co_scanned_cedar.pdf")
    print(f"wrote fixtures in {PDF_DIR}")


def build_text_pdf(lines: list[str], *, lines_per_page: int = 16) -> bytes:
    """Minimal Helvetica PDF. One text line per PDF text object line."""

    pages: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        wrapped = _wrap(line, 95) or [""]
        for piece in wrapped:
            if len(current) >= lines_per_page:
                pages.append(current)
                current = []
            current.append(piece)
    if current:
        pages.append(current)

    page_count = len(pages)
    # 1 catalog, 2 pages, 3 font, then a page object and a content object each.
    page_numbers = [4 + (2 * index) for index in range(page_count)]
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            f"<< /Type /Pages /Count {page_count} /Kids ["
            + " ".join(f"{number} 0 R" for number in page_numbers)
            + "] >>"
        ).encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for lines_on_page in pages:
        content = _page_stream(lines_on_page)
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents "
            + f"{len(objects) + 2} 0 R >>".encode()
        )
        objects.append(
            f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream"
        )
    return _assemble(objects)


def render_scan(path: Path) -> None:
    """Image-only PDF. Border specks only, so glyphs stay readable to OCR."""

    width, height = 1700, 2200
    image = Image.new("RGB", (width, height), (246, 243, 236))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(FONT), 42)
    title = ImageFont.truetype(str(FONT_BOLD), 54)
    y = 120
    for index, line in enumerate(SCAN_LINES):
        face = title if index == 0 else font
        draw.text((120, y), line, fill=(20, 20, 20), font=face)
        y += 78 if index == 0 else 64
    pixels = image.load()
    rng = random.Random(6)
    for _ in range(2500):
        x = rng.randrange(width)
        y_pixel = rng.randrange(height)
        if 80 < x < width - 80 and 80 < y_pixel < height - 80:
            continue
        pixels[x, y_pixel] = (210, 206, 196)
    image.save(path, "PDF", resolution=150.0)


def _wrap(line: str, width: int) -> list[str]:
    if len(line) <= width:
        return [line]
    words = line.split()
    rows: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                rows.append(current)
            current = word
    if current:
        rows.append(current)
    return rows


def _page_stream(lines: list[str]) -> bytes:
    commands = ["BT", "/F1 11 Tf", "54 750 Td"]
    for index, line in enumerate(lines):
        if index:
            commands.append("0 -14 Td")
        commands.append(f"({_escape(line)}) Tj")
    commands.append("ET")
    return "\n".join(commands).encode("latin-1", errors="replace")


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _assemble(objects: list[bytes]) -> bytes:
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode())
        output.extend(obj)
        if not obj.endswith(b"\n"):
            output.extend(b"\n")
        output.extend(b"endobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(output)


if __name__ == "__main__":
    main()
