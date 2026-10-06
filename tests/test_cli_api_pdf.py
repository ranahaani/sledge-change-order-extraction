import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from change_order_extract.api import app
from change_order_extract.cli import main
from change_order_extract.pipeline import extract_path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
CLEAN = SAMPLES / "co_01_aia_clean.txt"
INCONSISTENT = SAMPLES / "co_03_inconsistent_total.txt"
TEXT_PDF = SAMPLES / "pdf" / "co_01_aia_clean.pdf"
SCAN_PDF = SAMPLES / "pdf" / "co_scanned_cedar.pdf"


def test_cli_extract_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "out.json"
    assert main(["extract", str(CLEAN), "-o", str(output), "--provider", "rules"]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["change_order_number"]["value"] == "07"
    assert payload["total"]["value"] == "6345.13"
    assert payload["method"] == "rules"


def test_cli_rejects_unknown_suffix() -> None:
    assert main(["extract", str(ROOT / "pyproject.toml")]) == 2


def test_cli_eval_prints_the_summary(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["eval", "--samples", str(SAMPLES)]) == 0
    out = capsys.readouterr().out
    assert "labeled field accuracy: 1.000 (264/264)" in out
    assert "ECE: 0.108" in out


def test_api_extracts_text_and_rejects_unknown_uploads() -> None:
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    response = client.post("/extract", json={"text": INCONSISTENT.read_text(encoding="utf-8")})
    assert response.status_code == 200
    body = response.json()
    assert Decimal(str(body["total"]["value"])) == Decimal("2800.00")
    assert body["total"]["needs_review"] is True
    assert any(flag.startswith("math:") for flag in body["review_flags"])

    upload = client.post(
        "/extract/file",
        files={"file": ("co_01_aia_clean.txt", CLEAN.read_bytes(), "text/plain")},
    )
    assert upload.status_code == 200
    assert upload.json()["change_order_number"]["value"] == "07"

    rejected = client.post(
        "/extract/file",
        files={"file": ("notes.docx", b"not a pdf", "application/octet-stream")},
    )
    assert rejected.status_code == 415


def test_text_pdf_is_read_without_ocr() -> None:
    result = extract_path(TEXT_PDF)
    assert result.ocr_used is False
    assert result.page_count == 2
    assert result.change_order_number.value == "07"
    assert result.total.value == Decimal("6345.13")
    assert len(result.line_items) == 3


def test_scanned_pdf_uses_ocr() -> None:
    if shutil.which("tesseract") is None:
        pytest.skip("tesseract is not installed")
    result = extract_path(SCAN_PDF)
    assert result.ocr_used is True
    assert result.page_count == 1
    assert result.project_name.value is not None
    assert "Cedar" in result.project_name.value
    assert result.change_order_number.value == "6"
    assert result.total.value == Decimal("995.00")
    assert result.date.value is not None
    assert result.date.value.isoformat() == "2026-08-08"
