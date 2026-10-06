"""Behavioral checks that should hold even if fixture wording shifts slightly."""

from decimal import Decimal
from pathlib import Path

from change_order_extract.pipeline import extract_text
from change_order_extract.rules import RuleExtractor
from change_order_extract.schema import Scored

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def _read(name: str) -> str:
    return (SAMPLES / name).read_text(encoding="utf-8")


def test_stated_total_is_kept_when_line_items_do_not_sum() -> None:
    result = extract_text(_read("co_03_inconsistent_total.txt"))
    assert result.total.value == Decimal("2800.00")
    assert result.line_subtotal.value == Decimal("2800.00")
    assert result.total.needs_review
    assert result.line_subtotal.needs_review
    assert any("2564.00" in flag and "2800.00" in flag for flag in result.review_flags)
    assert all(item.amount.value != Decimal("2564.00") for item in result.line_items)


def test_ocr_digit_error_is_not_silently_corrected() -> None:
    result = extract_text(_read("co_05_ocr_noise.txt"))
    assert result.total.value == Decimal("4008.50")
    assert result.total.needs_review
    assert any(flag.startswith("math:") for flag in result.review_flags)
    assert result.warnings[0].startswith("ocr_normalization:")


def test_margin_note_is_not_added_into_the_total() -> None:
    result = extract_text(_read("co_06_margin_note.txt"))
    assert result.total.value == Decimal("8488.80")
    assert result.schedule_impact_days.value == 6
    assert result.schedule_impact_days.needs_review
    assert any("450.00" in (note.value or "") for note in result.unincorporated_notes)
    assert any(flag.startswith("note:") for flag in result.review_flags)
    assert any(flag.startswith("conflict:") for flag in result.review_flags)


def test_markup_on_a_credit_is_flagged_not_rewritten() -> None:
    result = extract_text(_read("co_08_credit.txt"))
    assert result.total.value == Decimal("-1322.00")
    assert result.markup.value == Decimal("98.00")
    assert result.markup.needs_review
    assert result.line_items[0].amount.value == Decimal("-2400.00")


def test_percent_without_a_dollar_amount_is_not_invented() -> None:
    result = extract_text(_read("co_09_prose_items.txt"))
    assert result.tax.value is None
    assert result.tax_percent.value == Decimal("7")
    assert result.total.value == Decimal("3654.05")
    assert result.total.needs_review
    assert result.line_items[0].unit.value == "SF"
    assert result.line_items[0].amount.value == Decimal("2015.00")


def test_missing_identity_fields_are_review_flags_not_guesses() -> None:
    result = extract_text(_read("co_04_missing_fields.txt"))
    assert result.project_name.value is None
    assert result.project_name.needs_review
    assert result.date.value is None
    assert result.owner.value is None
    assert result.owner.needs_review is False
    assert result.total.value == Decimal("518.00")


def test_ungrounded_value_is_downgraded() -> None:
    source = _read("co_01_aia_clean.txt")

    class _Hallucinating:
        name = "openai"

        def extract(self, text: str):
            draft = RuleExtractor().extract(text)
            draft.project_name = Scored(value="Hallucinated Tower", confidence=0.8)
            return draft

    result = extract_text(source, extractor=_Hallucinating())
    assert result.project_name.value == "Hallucinated Tower"
    assert result.project_name.needs_review
    assert result.project_name.confidence < 0.4
    assert any(flag == "grounding: project_name" for flag in result.review_flags)


def test_llm_failure_falls_back_to_rules() -> None:
    class _Boom:
        name = "openai"

        def extract(self, text: str):
            raise RuntimeError("timeout")

    result = extract_text(_read("co_01_aia_clean.txt"), extractor=_Boom())
    assert result.method == "rules_fallback"
    assert result.change_order_number.value == "07"
    assert any(warning.startswith("llm_failed:") for warning in result.warnings)


def test_empty_input_does_not_guess() -> None:
    result = extract_text("see attached")
    assert result.total.value is None
    assert result.document_confidence < 0.2
    assert "missing: total" in result.review_flags
