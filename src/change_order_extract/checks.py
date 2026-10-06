"""Deterministic checks that adjust confidence after extraction.

The provider (rules or LLM) proposes values and a starting confidence.
This module never overwrites a stated value with a recomputed one. It
only lowers confidence, raises it slightly when a check passes, and
sets ``needs_review``.

Order of adjustments, per field:

1. Ground the value in the source text. If it is not there, multiply
   confidence by 0.3 and force review. If it is there, add 0.03.
2. Line math: when quantity, unit price, and amount are all present,
   add 0.04 if ``qty * unit_price`` matches the amount within $0.05,
   otherwise multiply those three confidences by 0.55.
3. If the line-item amounts do not sum to ``line_subtotal``, multiply
   the subtotal by 0.5 and the total by 0.65.
4. If ``line_subtotal + markup + tax + other_fees`` does not match
   ``total`` (missing optional components count as zero), multiply the
   total by 0.5.
5. If a stated markup or tax percent does not reproduce the stated
   amount on either the line subtotal or the subtotal plus markup and
   fees, multiply that pair by 0.6.
6. If the text mentions more than one schedule duration, multiply the
   schedule confidence by 0.55.
7. Any present field under 0.65 confidence is flagged ``needs_review``.
   Missing required fields are flagged even though nothing was guessed.
8. Document confidence is the weighted mean of field confidences.
   Required fields that are missing contribute 0. One math failure
   multiplies the document score by 0.85. A schedule conflict or an
   unincorporated margin amount multiplies it by 0.9 when the arithmetic
   still ties. The score is capped at 0.93 so the pipeline never reports
   certainty.

Confirmed field confidence is also capped at 0.93.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal

from change_order_extract.schema import ChangeOrderExtraction, LineItem, Scored
from change_order_extract.textutil import (
    UNIT_SYNONYMS,
    grounded_date,
    grounded_identifier,
    grounded_money,
    grounded_string,
    quote_in_source,
    schedule_mentions,
)

REVIEW_THRESHOLD = 0.65
CONF_CAP = 0.93
MONEY_TOLERANCE = Decimal("0.05")
MATH_DOCUMENT_FACTOR = 0.85

# Weights for document confidence. Required fields stay in the average
# even when missing, so a blank date or total pulls the score down.
# Optional fields are omitted when they were not on the document.
FIELD_WEIGHTS: dict[str, float] = {
    "change_order_number": 1.5,
    "total": 1.5,
    "line_subtotal": 1.2,
    "description": 1.0,
    "project_name": 1.0,
    "contractor": 0.9,
    "date": 0.8,
    "owner": 0.8,
    "project_number": 0.8,
    "schedule_impact_days": 0.7,
    "markup": 0.6,
    "tax": 0.6,
    "reason": 0.6,
    "approval_status": 0.5,
    "markup_percent": 0.4,
    "tax_percent": 0.4,
    "other_fees": 0.4,
    "currency": 0.3,
}
REQUIRED_FOR_SCORE = {
    "project_name",
    "change_order_number",
    "contractor",
    "description",
    "total",
    "date",
}
REQUIRED_REVIEW = REQUIRED_FOR_SCORE | {"line_items"}

_ZERO_PHRASES = (
    "tax exempt",
    "no tax",
    "no markup",
    "markup: none",
    "without markup",
)


def apply_checks(result: ChangeOrderExtraction, source: str) -> ChangeOrderExtraction:
    """Return a copy of ``result`` with confidence and review flags adjusted."""

    updated = result.model_copy(deep=True)
    flags: list[str] = []
    grounded: set[int] = set()

    for name, kind in _SCALAR_KINDS:
        field = getattr(updated, name)
        field, ok = _ground_field(field, kind, source)
        setattr(updated, name, field)
        if field.value is not None:
            grounded.add(id(field))
            if not ok:
                flags.append(f"grounding: {name}")

    new_items: list[LineItem] = []
    for index, item in enumerate(updated.line_items):
        item, item_flags = _check_line_item(item, source, index)
        new_items.append(item)
        flags.extend(item_flags)
    updated.line_items = new_items

    flags.extend(_check_rollup(updated))
    flags.extend(_check_percents(updated))
    flags.extend(_check_schedule_conflict(updated, source))
    flags.extend(_flag_missing(updated))

    for name, _kind in _SCALAR_KINDS:
        setattr(updated, name, _finalize_field(getattr(updated, name)))
    finalized_items: list[LineItem] = []
    for item in updated.line_items:
        finalized_items.append(
            item.model_copy(
                update={
                    "description": _finalize_field(item.description),
                    "quantity": _finalize_field(item.quantity),
                    "unit": _finalize_field(item.unit),
                    "unit_price": _finalize_field(item.unit_price),
                    "amount": _finalize_field(item.amount),
                }
            )
        )
    updated.line_items = finalized_items
    updated.unincorporated_notes = [_finalize_field(note) for note in updated.unincorporated_notes]

    math_failure = any(flag.startswith("math:") for flag in flags)
    soft_failure = any(flag.startswith(("note:", "conflict:")) for flag in flags)
    updated.review_flags = _unique(flags)
    updated.document_confidence = _document_confidence(
        updated, math_failure=math_failure, soft_failure=soft_failure
    )
    return updated


_SCALAR_KINDS: tuple[tuple[str, str], ...] = (
    ("project_name", "text"),
    ("project_number", "id"),
    ("change_order_number", "id"),
    ("date", "date"),
    ("contractor", "text"),
    ("owner", "text"),
    ("description", "text"),
    ("currency", "currency"),
    ("line_subtotal", "money"),
    ("markup", "money"),
    ("markup_percent", "percent"),
    ("tax", "money"),
    ("tax_percent", "percent"),
    ("other_fees", "money"),
    ("total", "money"),
    ("schedule_impact_days", "days"),
    ("reason", "text"),
    ("approval_status", "approval"),
)


def _ground_field(field: Scored[object], kind: str, source: str) -> tuple[Scored[object], bool]:
    if field.value is None:
        return field, True
    ok = _is_grounded(kind, field.value, field.evidence, source)
    if ok:
        return _with_confidence(field, field.confidence + 0.03), True
    return _with_confidence(field, field.confidence * 0.3, review=True), False


def _is_grounded(kind: str, value: object, evidence: str | None, source: str) -> bool:
    quoted = quote_in_source(evidence, source)
    if kind == "money":
        amount = value if isinstance(value, Decimal) else Decimal(str(value))
        if grounded_money(amount, source):
            return True
        if quoted and evidence is not None and grounded_money(amount, evidence):
            return True
        if amount == 0 and any(phrase in source.casefold() for phrase in _ZERO_PHRASES):
            return True
        return False
    if kind == "date":
        if not isinstance(value, date):
            return False
        return grounded_date(value, source) or (
            quoted and evidence is not None and grounded_date(value, evidence)
        )
    if kind == "days":
        return _grounded_days(int(str(value)), source)
    if kind == "percent":
        number = format(Decimal(str(value)).normalize(), "f")
        return re_search_percent(number, source) or (
            quoted and evidence is not None and re_search_percent(number, evidence)
        )
    if kind == "currency":
        return (value == "USD" and "$" in source) or str(value).casefold() in source.casefold()
    if kind == "approval":
        return _approval_grounded(str(value), source)
    if kind == "id":
        text = str(value)
        if quoted and evidence is not None and grounded_identifier(text, evidence):
            return True
        return grounded_identifier(text, source)
    text = str(value)
    if grounded_string(text, source):
        return True
    return bool(quoted and evidence is not None and grounded_string(text, evidence))


def re_search_percent(number: str, source: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(number)}\s*%", source) is not None


def _grounded_days(value: int, source: str) -> bool:
    if value == 0 and re.search(r"no schedule|increase of\s+0|\b0\s+days", source, re.I):
        return True
    if re.search(rf"(?<!\d){abs(value)}(?!\d)", source):
        return True
    words = {1: "one", 2: "two", 3: "three"}
    word = words.get(abs(value))
    return bool(word and re.search(rf"\b{word}\b", source, re.I))


def _approval_grounded(value: str, source: str) -> bool:
    low = source.casefold()
    table = {
        "approved": ("approved", "signed"),
        "pending": ("pending", "not signed", "waiting", "blank", "____", "still need"),
        "draft": ("draft",),
        "rejected": ("rejected", "denied"),
        "unknown": (),
    }
    return any(token in low for token in table.get(value, (value.casefold(),)))


def _check_line_item(item: LineItem, source: str, index: int) -> tuple[LineItem, list[str]]:
    flags: list[str] = []
    description, desc_ok = _ground_field(item.description, "text", source)
    quantity, _qty_ok = _ground_field(item.quantity, "money", source)
    unit = _ground_unit(item.unit, source)
    unit_price, _price_ok = _ground_field(item.unit_price, "money", source)
    amount, amount_ok = _ground_field(item.amount, "money", source)
    if not desc_ok and description.value is not None:
        flags.append(f"grounding: line_items[{index}].description")
    if not amount_ok and amount.value is not None:
        flags.append(f"grounding: line_items[{index}].amount")

    if quantity.value is not None and unit_price.value is not None and amount.value is not None:
        expected = quantize(quantity.value * unit_price.value)
        if abs(expected - amount.value) <= MONEY_TOLERANCE:
            quantity = _with_confidence(quantity, quantity.confidence + 0.04)
            unit_price = _with_confidence(unit_price, unit_price.confidence + 0.04)
            amount = _with_confidence(amount, amount.confidence + 0.04)
        else:
            quantity = _with_confidence(quantity, quantity.confidence * 0.55, review=True)
            unit_price = _with_confidence(unit_price, unit_price.confidence * 0.55, review=True)
            amount = _with_confidence(amount, amount.confidence * 0.55, review=True)
            flags.append(
                f"math: line_items[{index}] {quantity.value} x {unit_price.value} "
                f"!= {amount.value}"
            )
    return (
        item.model_copy(
            update={
                "description": description,
                "quantity": quantity,
                "unit": unit,
                "unit_price": unit_price,
                "amount": amount,
            }
        ),
        flags,
    )


def _ground_unit(field: Scored[str], source: str) -> Scored[str]:
    if field.value is None:
        return field
    haystack = field.evidence if quote_in_source(field.evidence, source) else source
    folded = haystack.casefold()
    code = field.value.upper()
    if code.casefold() in folded:
        return _with_confidence(field, field.confidence + 0.03)
    for synonym in UNIT_SYNONYMS.get(code, ()):
        if synonym in folded:
            return _with_confidence(field, field.confidence + 0.03)
    return _with_confidence(field, field.confidence * 0.3, review=True)


def _check_rollup(result: ChangeOrderExtraction) -> list[str]:
    flags: list[str] = []
    amounts = [item.amount.value for item in result.line_items if item.amount.value is not None]
    line_sum = sum(amounts, Decimal("0")) if amounts else None
    subtotal = result.line_subtotal.value
    if line_sum is not None and subtotal is not None and abs(line_sum - subtotal) > MONEY_TOLERANCE:
        result.line_subtotal = _with_confidence(
            result.line_subtotal, result.line_subtotal.confidence * 0.5, review=True
        )
        result.total = _with_confidence(result.total, result.total.confidence * 0.65, review=True)
        flags.append(f"math: line items sum to {line_sum:.2f} but line_subtotal is {subtotal:.2f}")

    base = subtotal
    if base is None and line_sum is not None:
        base = line_sum
    total = result.total.value
    if base is not None and total is not None:
        expected = base + _or_zero(result.markup) + _or_zero(result.tax) + _or_zero(result.other_fees)
        expected = quantize(expected)
        if abs(expected - total) > MONEY_TOLERANCE:
            result.total = _with_confidence(result.total, result.total.confidence * 0.5, review=True)
            flags.append(
                f"math: total {total:.2f} != {expected:.2f} from subtotal and stated adjustments"
            )
    return flags


def _check_percents(result: ChangeOrderExtraction) -> list[str]:
    flags: list[str] = []
    subtotal = result.line_subtotal.value
    if (
        subtotal is not None
        and result.markup.value is not None
        and result.markup_percent.value is not None
    ):
        expected = quantize(subtotal * result.markup_percent.value / Decimal("100"))
        if abs(expected - result.markup.value) > MONEY_TOLERANCE:
            result.markup = _with_confidence(result.markup, result.markup.confidence * 0.6, review=True)
            result.markup_percent = _with_confidence(
                result.markup_percent, result.markup_percent.confidence * 0.6, review=True
            )
            flags.append(
                f"math: markup {result.markup.value:.2f} is not "
                f"{result.markup_percent.value}% of line_subtotal"
            )
    if (
        subtotal is not None
        and result.tax.value is not None
        and result.tax_percent.value is not None
    ):
        bases = [subtotal, subtotal + _or_zero(result.markup) + _or_zero(result.other_fees)]
        expecteds = [quantize(base * result.tax_percent.value / Decimal("100")) for base in bases]
        if all(abs(expected - result.tax.value) > MONEY_TOLERANCE for expected in expecteds):
            result.tax = _with_confidence(result.tax, result.tax.confidence * 0.6, review=True)
            result.tax_percent = _with_confidence(
                result.tax_percent, result.tax_percent.confidence * 0.6, review=True
            )
            flags.append(
                f"math: tax {result.tax.value:.2f} is not {result.tax_percent.value}% "
                "of the subtotal or the subtotal plus markup and fees"
            )
    return flags


def _check_schedule_conflict(result: ChangeOrderExtraction, source: str) -> list[str]:
    mentions = schedule_mentions(source)
    if len(mentions) <= 1 or result.schedule_impact_days.value is None:
        return []
    result.schedule_impact_days = _with_confidence(
        result.schedule_impact_days,
        result.schedule_impact_days.confidence * 0.55,
        review=True,
    )
    listed = ", ".join(str(value) for value in sorted(mentions))
    return [f"conflict: schedule mentions {listed} days"]


def _flag_missing(result: ChangeOrderExtraction) -> list[str]:
    flags: list[str] = []
    for name in ("project_name", "change_order_number", "contractor", "description", "total", "date"):
        field = getattr(result, name)
        if field.value is None:
            setattr(result, name, field.model_copy(update={"needs_review": True}))
            flags.append(f"missing: {name}")
    if not result.line_items:
        flags.append("missing: line_items")
    for note in result.unincorporated_notes:
        if note.value:
            flags.append(f"note: {note.value}")
    return flags


def _document_confidence(
    result: ChangeOrderExtraction, *, math_failure: bool, soft_failure: bool
) -> float:
    weighted = 0.0
    weight_sum = 0.0
    for name, weight in FIELD_WEIGHTS.items():
        field = getattr(result, name)
        if field.value is None and name not in REQUIRED_FOR_SCORE:
            continue
        weighted += field.confidence * weight
        weight_sum += weight
    if result.line_items:
        for item in result.line_items:
            weighted += item.amount.confidence * 0.8
            weight_sum += 0.8
    else:
        weight_sum += 1.0
    if weight_sum == 0:
        score = 0.0
    else:
        score = weighted / weight_sum
    if math_failure:
        score *= MATH_DOCUMENT_FACTOR
    elif soft_failure:
        # A margin note or conflicting duration does not break the arithmetic,
        # but the document is not clean enough to keep the full score.
        score *= 0.9
    return round(min(CONF_CAP, max(0.0, score)), 3)


def _or_zero(field: Scored[Decimal]) -> Decimal:
    return field.value if field.value is not None else Decimal("0")


def quantize(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"))


def _with_confidence(field: Scored[object], confidence: float, *, review: bool = False) -> Scored[object]:
    conf = round(min(CONF_CAP, max(0.0, confidence)), 3)
    needs = field.needs_review or review
    return field.model_copy(update={"confidence": conf, "needs_review": needs})


def _finalize_field(field: Scored[object]) -> Scored[object]:
    conf = round(min(CONF_CAP, max(0.0, field.confidence)), 3)
    review = field.needs_review or (field.value is not None and conf < REVIEW_THRESHOLD)
    evidence = field.evidence
    if evidence is not None:
        evidence = " ".join(evidence.split())
        if len(evidence) > 300:
            evidence = evidence[:299] + "..."
    return field.model_copy(update={"confidence": conf, "needs_review": review, "evidence": evidence})


def _unique(flags: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for flag in flags:
        if flag not in seen:
            seen.add(flag)
            ordered.append(flag)
    return ordered
