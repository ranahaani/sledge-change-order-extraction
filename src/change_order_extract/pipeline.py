"""Orchestrate load, extract, and deterministic checks."""

from __future__ import annotations

from pathlib import Path

from change_order_extract.checks import apply_checks
from change_order_extract.pdf_text import load_bytes, load_path
from change_order_extract.providers import Extractor, build_extractor
from change_order_extract.rules import RuleExtractor
from change_order_extract.schema import ChangeOrderExtraction
from change_order_extract.textutil import normalize_ocr_text


def extract_text(
    text: str,
    *,
    provider: str = "auto",
    source_name: str | None = None,
    extractor: Extractor | None = None,
    ocr_used: bool = False,
    page_count: int | None = None,
    warnings: list[str] | None = None,
) -> ChangeOrderExtraction:
    """Extract a change order from plain text.

    ``provider`` is ``auto`` (OpenAI-compatible if ``OPENAI_API_KEY`` is set,
    otherwise rules), ``rules``, or ``openai``. Pass ``extractor`` to inject
    a fake in tests.
    """

    prepared, changes = normalize_ocr_text(text)
    notes = list(warnings or [])
    if changes:
        notes.append(f"ocr_normalization:{changes}")
    if extractor is None:
        extractor = build_extractor(provider)
    method = extractor.name
    try:
        draft = extractor.extract(prepared)
    except Exception as exc:
        if extractor.name == "rules":
            raise
        notes.append(f"llm_failed:{exc.__class__.__name__}")
        draft = RuleExtractor().extract(prepared)
        method = "rules_fallback"
    checked = apply_checks(draft, prepared)
    return checked.model_copy(
        update={
            "method": method,
            "source_name": source_name if source_name is not None else checked.source_name,
            "ocr_used": ocr_used,
            "page_count": page_count,
            "warnings": [*notes, *draft.warnings],
        }
    )


def extract_path(path: str | Path, *, provider: str = "auto") -> ChangeOrderExtraction:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    loaded = load_path(file_path)
    return extract_text(
        loaded.text,
        provider=provider,
        source_name=file_path.name,
        ocr_used=loaded.ocr_used,
        page_count=loaded.page_count,
        warnings=loaded.warnings,
    )


def extract_bytes(
    data: bytes,
    suffix: str,
    *,
    provider: str = "auto",
    source_name: str | None = None,
) -> ChangeOrderExtraction:
    loaded = load_bytes(data, suffix.lower())
    return extract_text(
        loaded.text,
        provider=provider,
        source_name=source_name,
        ocr_used=loaded.ocr_used,
        page_count=loaded.page_count,
        warnings=loaded.warnings,
    )
