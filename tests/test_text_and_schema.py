from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from change_order_extract.schema import Scored
from change_order_extract.textutil import (
    find_money,
    normalize_ocr_text,
    parse_date,
    parse_decimal,
)


def test_confidence_must_be_between_zero_and_one() -> None:
    with pytest.raises(ValidationError):
        Scored[str](value="x", confidence=1.2)


def test_money_parses_credits_parentheses_and_skips_percents() -> None:
    assert parse_decimal("($1,322.00)") == Decimal("-1322.00")
    assert parse_decimal("(1,420.00)") == Decimal("-1420.00")
    assert find_money("Sales Tax (8.25%) $483.58") == [Decimal("483.58")]
    assert find_money("No markup") == []


def test_date_parses_common_change_order_formats() -> None:
    assert parse_date("March 14, 2025") == date(2025, 3, 14)
    assert parse_date("02/02/2026") == date(2026, 2, 2)
    assert parse_date("Issued: 2026-06-01") == date(2026, 6, 1)
    assert parse_date("not a date") is None


def test_ocr_normalization_repairs_letters_and_leaves_real_digits() -> None:
    raw = "PR0JECT: LARKSPUR RES1DENCE T0WER\nDATE: Apr1l 2, 2026\nT0TAL 4,008.50\n2,450.OO"
    text, changes = normalize_ocr_text(raw)
    assert "PROJECT" in text
    assert "RESIDENCE" in text
    assert "TOWER" in text
    assert "April" in text
    assert "TOTAL" in text
    assert "4,008.50" in text
    assert "2,450.00" in text
    assert changes >= 4
