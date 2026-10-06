"""Rule extractor for change orders.

This is the default provider. It does not call a network and it does not
invent amounts that are not written down. Prose coverage is limited to a
few common phrasings; anything it is unsure about stays low-confidence.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal

from change_order_extract.schema import ChangeOrderExtraction, LineItem, Scored
from change_order_extract.textutil import (
    find_money,
    find_percent,
    norm_ws,
    normalize_ocr_text,
    parse_date,
    parse_decimal,
    quantize_money,
)

_UNIT = r"EA|LF|SF|SY|CY|LS|HR|LB|TON|TN|GAL|DAYS|DAY|PCS|PC"
_MONEY_TOL = Decimal("0.05")

_DOT_AT = re.compile(
    rf"^(?:[-*]\s*)?(?P<desc>.+?)\s*\.{{2,}}\s*"
    rf"(?P<qty>\d[\d,]*(?:\.\d+)?)\s+"
    rf"(?P<unit>{_UNIT})\s*@\s*"
    rf"\$?(?P<price>\d[\d,]*(?:\.\d+)?)\s*=\s*"
    rf"\$?(?P<amount>\d[\d,]*(?:\.\d+)?)\s*$",
    re.I,
)
_SLASH = re.compile(
    rf"^(?:[-*]\s*)?(?P<desc>.+?)\s*\.{{2,}}\s*"
    rf"(?P<qty>\d[\d,]*(?:\.\d+)?)\s+"
    rf"(?P<unit>{_UNIT})\s+"
    rf"\$?(?P<price>\d[\d,]*(?:\.\d+)?)\s*/\s*[A-Za-z.]+\s+"
    rf"\$?(?P<amount>\d[\d,]*(?:\.\d+)?)\s*$",
    re.I,
)
_COLS = re.compile(
    rf"^(?:[-*]\s*)?(?P<desc>.+?)\s+"
    rf"(?P<qty>-?\d[\d,]*(?:\.\d+)?)\s+"
    rf"(?P<unit>{_UNIT})\s+"
    rf"\$?(?P<price>-?\$?\d[\d,]*(?:\.\d+)?)\s+"
    rf"\$?(?P<amount>-?\$?\d[\d,]*(?:\.\d+)?)\s*$",
    re.I,
)
_DESC_LABEL = re.compile(
    r"^(?:the contract is changed as follows|description of work|description|desc|scope)"
    r"\s*(?::\s*|\s*$)(.*)$",
    re.I,
)
_STOP_LABEL = re.compile(
    r"^(?:project|job|change order|date|issued|owner|contractor|client|from|to|cause|"
    r"reason|category|status|approval|signature|schedule|subtotal|total|qty|item)\b",
    re.I,
)
_CO_PATTERNS = (
    re.compile(
        r"\b(?:change\s+order|c\.?o\.?)\s*(?:number|no\.?|#)\s*[:.]?\s*#?\s*"
        r"((?=[A-Z0-9.-]*\d)[A-Z0-9][\w.-]*)",
        re.I,
    ),
    re.compile(r"\b(COR-\d+)\b", re.I),
    re.compile(r"\b(CO-\d{4}-\d+)\b", re.I),
    re.compile(r"\bPCO\s*-?\s*#?\s*(\d+)\b", re.I),
    re.compile(r"\bchange\s+order\s+#?\s*(\d+)\b", re.I),
    re.compile(r"\bCO\s+#\s*(\d+)\b", re.I),
)
_PROJECT_NO = (
    re.compile(
        r"^[ \t]*(?:project|job)\s*(?:no\.?|number|#)\s*[:.]?\s*([A-Z0-9][\w.-]+)",
        re.I | re.M,
    ),
    re.compile(r"\(\s*job\s+([A-Z0-9][\w.-]+)\s*\)", re.I),
    re.compile(r"\(\s*no\.?\s+([A-Z0-9][\w.-]+)\s*\)", re.I),
    re.compile(r"\(\s*project\s+([A-Z0-9][\w.-]+)\s*\)", re.I),
    re.compile(r"\bproject\s+([A-Z]{1,8}-\d[\w.-]*)", re.I),
)
_PROJECT_NAME = (
    re.compile(r"^[ \t]*project(?:\s*name)?\s*:\s*(.+?)\s*$", re.I | re.M),
    re.compile(r"^[ \t]*job\s*:\s*(.+?)\s*$", re.I | re.M),
    re.compile(r"for the\s+(.+?)\s*\(\s*(?:job|project)\b", re.I),
)
_DATE_LABELS = (
    re.compile(r"^[ \t]*date\s*:\s*(.+)$", re.I | re.M),
    re.compile(r"^[ \t]*issued\s*:\s*(.+)$", re.I | re.M),
    re.compile(r"\bdated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})", re.I),
)
_OWNER = (
    (re.compile(r"^[ \t]*to\s*\(\s*owner\s*\)\s*:\s*(.+)$", re.I | re.M), 0.9),
    (re.compile(r"^[ \t]*(?:owner|client)\s*:\s*(.+)$", re.I | re.M), 0.9),
    (re.compile(r"\bowner is (?:the\s+)?(.+?)(?:\.|,)", re.I), 0.82),
)
_CONTRACTOR = (
    re.compile(r"^[ \t]*contractor\s*:\s*(.+)$", re.I | re.M),
    re.compile(r"^[ \t]*from\s*\(\s*contractor\s*\)\s*:\s*(.+)$", re.I | re.M),
)
_REASON = (
    re.compile(r"^[ \t]*reason(?:\s*code)?\s*:\s*(.+)$", re.I | re.M),
    re.compile(r"^[ \t]*category\s*:\s*(.+)$", re.I | re.M),
    re.compile(r"\bcause(?:\s+is|\s*:)\s*([^\n.]+)", re.I),
    re.compile(r"\bthis is an?\s+([^\n.]+)", re.I),
)
_SCHEDULE = (
    (re.compile(r"schedule\s+impact\s*:\s*(-?\d+)\s*(?:calendar\s+|working\s+)?days?", re.I), 0.9),
    (re.compile(r"\btime\s*:\s*(-?\d+)\s+days?", re.I), 0.86),
    (re.compile(r"\bincrease of\s+(-?\d+)\s+days?", re.I), 0.86),
    (re.compile(r"\badds?\s+(-?\d+)\s+(?:calendar\s+|working\s+)?days?", re.I), 0.84),
    (re.compile(r"\badds?\s+(one|two|three)\s+days?", re.I), 0.8),
    (re.compile(r"\bno schedule (?:hit|impact|change)\b", re.I), 0.84),
)
_WORD_NUM = {"one": 1, "two": 2, "three": 3}
_PROSE_MEASURED = re.compile(
    r"(?:about|approx(?:imately)?|roughly)?\s*"
    r"(?P<qty>\d[\d,]*)\s+"
    r"(?P<unit>square feet|sq\.?\s*ft\.?|linear feet|hours?|each|SF|LF|EA|HR|SY|LS)\s+"
    r"at\s+\$?(?P<price>\d[\d,]*(?:\.\d+)?)"
    r"(?:[^.]{0,80}?comes to\s+\$?(?P<amount>\d[\d,]*(?:\.\d+)?))?",
    re.I,
)
_PROSE_LUMP = re.compile(
    r"(?:plus|and)\s+(?:a|an)\s+(?P<desc>[A-Za-z][\w'&/ -]{2,40}?)\s+at\s+"
    r"\$?(?P<amount>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)",
    re.I,
)
_LEADING = re.compile(r"^(?:(?:please|approve|an|a|extra|for|the|we|need|to|add)\s+)+", re.I)
_WORK = re.compile(
    r"\b(furnish|install|patch|relocate|extend|replace|delete|add|inject|recoat|"
    r"remove|provide|substitute|demo)\b",
    re.I,
)
_COMPANY = re.compile(
    r"\b(Inc\.?|LLC|LLP|Contractors?|Builders?|Construction|Mechanical|Roofing|"
    r"Concrete|Electric|Civil|Restoration|Millwork|Glazing|Interiors)\b",
    re.I,
)
_TOTAL_RANK = {"grand total": 0, "total change": 1, "total credit": 2, "total": 3}


class RuleExtractor:
    """Deterministic extractor used when no API key is configured."""

    name = "rules"

    def extract(self, text: str) -> ChangeOrderExtraction:
        normalized, _changes = normalize_ocr_text(text)
        lines = [line.rstrip() for line in normalized.splitlines()]
        items, consumed = _line_items(lines)
        if not items:
            items = _prose_items(normalized)
        line_sum = _sum_amounts(items)
        totals, money_lines = _totals(lines, consumed, line_sum)
        consumed |= money_lines
        notes = _notes(lines, consumed, items, totals)
        return ChangeOrderExtraction(
            project_name=_project_name(normalized),
            project_number=_project_number(normalized),
            change_order_number=_change_order_number(normalized),
            date=_date_field(normalized),
            contractor=_contractor(normalized, lines),
            owner=_owner(normalized),
            description=_description(lines, normalized),
            currency=_currency(normalized),
            line_items=items,
            line_subtotal=totals["line_subtotal"],
            markup=totals["markup"],
            markup_percent=totals["markup_percent"],
            tax=totals["tax"],
            tax_percent=totals["tax_percent"],
            other_fees=totals["other_fees"],
            total=totals["total"],
            schedule_impact_days=_schedule(normalized),
            reason=_reason(normalized),
            approval_status=_approval(normalized),
            unincorporated_notes=notes,
            method=self.name,
        )


def _line_items(lines: list[str]) -> tuple[list[LineItem], set[int]]:
    items: list[LineItem] = []
    consumed: set[int] = set()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or _is_header(stripped) or classify_money_line(stripped):
            continue
        parsed = _parse_item_line(stripped)
        if parsed is None:
            continue
        items.append(parsed)
        consumed.add(index)
    return items, consumed


def _is_header(line: str) -> bool:
    low = line.casefold()
    has_desc = "description" in low or low.startswith("item")
    has_qty = "qty" in low or "quantity" in low
    has_amount = any(token in low for token in ("amount", "ext", "price"))
    return has_desc and has_qty and has_amount and not re.search(r"\d", line)


def _parse_item_line(line: str) -> LineItem | None:
    match = _DOT_AT.match(line) or _SLASH.match(line) or _COLS.match(line)
    if not match:
        return None
    description = _clean_desc(match.group("desc"))
    if not description or not re.search(r"[A-Za-z]", description):
        return None
    quantity = parse_decimal(match.group("qty"))
    price = parse_decimal(match.group("price"))
    amount = parse_decimal(match.group("amount"))
    if quantity is None or price is None or amount is None:
        return None
    unit = _canonical_unit(match.group("unit"))
    return _item(
        description,
        quantity,
        unit,
        quantize_money(price),
        quantize_money(amount),
        line,
        0.86,
    )


def _prose_items(text: str) -> list[LineItem]:
    flat = norm_ws(text)
    items: list[LineItem] = []
    for match in _PROSE_MEASURED.finditer(flat):
        prefix = flat[max(0, match.start() - 90) : match.start()]
        clause = re.split(r"[.!;]", prefix)[-1]
        description = _trim_leading(clause.strip(" ,"))
        quantity = parse_decimal(match.group("qty"))
        price = parse_decimal(match.group("price"))
        amount_token = match.group("amount")
        amount = parse_decimal(amount_token) if amount_token else None
        if quantity is None or price is None or not description:
            continue
        if amount is None:
            amount = quantize_money(quantity * price)
        items.append(
            _item(
                description,
                quantity,
                _canonical_unit(match.group("unit")),
                quantize_money(price),
                quantize_money(amount),
                match.group(0),
                0.7,
            )
        )
    for match in _PROSE_LUMP.finditer(flat):
        description = _trim_leading(norm_ws(match.group("desc")))
        if any(description.casefold() in (item.description.value or "").casefold() for item in items):
            continue
        amount = parse_decimal(match.group("amount"))
        if amount is None or not description:
            continue
        items.append(
            LineItem(
                description=_scored(description, 0.6, match.group(0)),
                amount=_scored(quantize_money(amount), 0.6, match.group(0)),
            )
        )
    return items


def _totals(
    lines: list[str], consumed: set[int], line_sum: Decimal | None
) -> tuple[dict[str, Scored[Decimal]], set[int]]:
    subtotals: list[tuple[Decimal, str]] = []
    totals: list[tuple[str, Decimal, str]] = []
    markup: Scored[Decimal] = Scored()
    markup_percent: Scored[Decimal] = Scored()
    tax: Scored[Decimal] = Scored()
    tax_percent: Scored[Decimal] = Scored()
    other: Scored[Decimal] = Scored()
    money_lines: set[int] = set()
    for index, raw in enumerate(lines):
        if index in consumed:
            continue
        line = raw.strip()
        label = classify_money_line(line)
        if label is None:
            continue
        money_lines.add(index)
        amounts = find_money(line)
        amount = amounts[-1] if amounts else None
        percent = find_percent(line)
        low = line.casefold()
        if label == "markup":
            if re.search(r"\bno markup\b", low):
                markup = _scored(Decimal("0.00"), 0.86, line)
            elif amount is not None:
                markup = _scored(amount, 0.88, line)
            if percent is not None:
                markup_percent = _scored(percent, 0.86, line)
        elif label == "tax":
            if amount is not None:
                tax = _scored(amount, 0.88, line)
            elif re.search(r"exempt|no tax", low):
                tax = _scored(Decimal("0.00"), 0.86, line)
            if percent is not None:
                tax_percent = _scored(percent, 0.86, line)
        elif label == "other_fees" and amount is not None:
            other = _scored(amount, 0.86, line)
        elif label == "subtotal" and amount is not None:
            subtotals.append((amount, line))
        elif label in _TOTAL_RANK and amount is not None:
            totals.append((label, amount, line))
    return (
        {
            "line_subtotal": _choose_subtotal(subtotals, line_sum),
            "markup": markup,
            "markup_percent": markup_percent,
            "tax": tax,
            "tax_percent": tax_percent,
            "other_fees": other,
            "total": _choose_total(totals),
        },
        money_lines,
    )


def classify_money_line(line: str) -> str | None:
    low = line.casefold().strip()
    if not low:
        return None
    if re.search(r"\b(markup|overhead)\b", low) or "oh&p" in low or "o&p" in low.replace(" ", ""):
        return "markup"
    if re.search(r"\bbond\b", low):
        return "other_fees"
    if re.match(r"grand\s+total\b", low):
        return "grand total"
    if re.match(r"total\s+change\b", low):
        return "total change"
    if re.match(r"total\s+credit\b", low):
        return "total credit"
    if re.match(r"subtotal\b", low):
        return "subtotal"
    if re.search(r"\btax\b", low):
        return "tax"
    if re.match(r"net\b", low):
        return "subtotal"
    if re.match(r"total\b", low):
        return "total"
    return None


def _choose_subtotal(candidates: list[tuple[Decimal, str]], line_sum: Decimal | None) -> Scored[Decimal]:
    if not candidates:
        return Scored()
    if line_sum is None:
        amount, evidence = candidates[0]
        return _scored(amount, 0.8, evidence)
    amount, evidence = min(candidates, key=lambda item: abs(item[0] - line_sum))
    confidence = 0.9 if abs(amount - line_sum) <= _MONEY_TOL else 0.55
    return _scored(amount, confidence, evidence)


def _choose_total(candidates: list[tuple[str, Decimal, str]]) -> Scored[Decimal]:
    if not candidates:
        return Scored()
    best = min(_TOTAL_RANK[label] for label, _amount, _evidence in candidates)
    pool = [item for item in candidates if _TOTAL_RANK[item[0]] == best]
    _label, amount, evidence = pool[-1]
    return _scored(amount, 0.9, evidence)


def _notes(
    lines: list[str],
    consumed: set[int],
    items: list[LineItem],
    totals: dict[str, Scored[Decimal]],
) -> list[Scored[str]]:
    known: list[Decimal] = []
    for item in items:
        if item.amount.value is not None:
            known.append(item.amount.value)
        if item.unit_price.value is not None:
            known.append(item.unit_price.value)
    for field in totals.values():
        if field.value is not None:
            known.append(field.value)
    notes: list[Scored[str]] = []
    for index, raw in enumerate(lines):
        if index in consumed:
            continue
        line = raw.strip()
        if not line:
            continue
        unknown = [
            amount
            for amount in find_money(line)
            if not any(abs(amount - seen) <= _MONEY_TOL for seen in known)
        ]
        if unknown:
            rendered = ", ".join(f"{amount:.2f}" for amount in unknown)
            notes.append(_scored(f"unincorporated amount {rendered}: {norm_ws(line)}", 0.58, line))
    return notes


def _project_name(text: str) -> Scored[str]:
    for pattern in _PROJECT_NAME:
        match = pattern.search(text)
        if not match:
            continue
        value = _clean_party(match.group(1))
        if value and len(value) <= 120:
            confidence = 0.9 if ":" in match.group(0) else 0.74
            return _scored(value, confidence, match.group(0))
    return Scored()


def _project_number(text: str) -> Scored[str]:
    for pattern in _PROJECT_NO:
        match = pattern.search(text)
        if match and re.search(r"\d", match.group(1)):
            return _scored(match.group(1).strip(), 0.88, match.group(0))
    return Scored()


def _change_order_number(text: str) -> Scored[str]:
    for pattern in _CO_PATTERNS:
        match = pattern.search(text)
        if match:
            return _scored(match.group(1).strip(), 0.9, match.group(0))
    return Scored()


def _date_field(text: str) -> Scored[date]:
    for pattern in _DATE_LABELS:
        match = pattern.search(text)
        if not match:
            continue
        parsed = parse_date(match.group(1))
        if parsed is not None:
            return _scored(parsed, 0.9, match.group(0))
    return Scored()


def _owner(text: str) -> Scored[str]:
    for pattern, confidence in _OWNER:
        match = pattern.search(text)
        if not match:
            continue
        value = _clean_party(match.group(1))
        if value and not value.startswith("_") and len(value) <= 120:
            return _scored(value, confidence, match.group(0))
    return Scored()


def _contractor(text: str, lines: list[str]) -> Scored[str]:
    for pattern in _CONTRACTOR:
        match = pattern.search(text)
        if match:
            value = _clean_party(match.group(1))
            if value:
                return _scored(value, 0.9, match.group(0))
    named = None
    evidence = None
    submitting = re.search(r"([A-Z][A-Za-z0-9&.,' -]{2,50}?)\s+is submitting", text)
    if submitting:
        named = _clean_party(submitting.group(1))
        evidence = submitting.group(0)
    else:
        we_are = re.search(
            r"\bwe(?:'re| are)\s+([A-Za-z][A-Za-z0-9&.,' -]{2,40}?)(?:\.|,)",
            text,
            re.I,
        )
        if we_are:
            named = _clean_party(we_are.group(1))
            evidence = we_are.group(0)
    signature = _company_line(lines)
    if named and signature and _shares_token(named, signature):
        chosen = signature if len(signature) >= len(named) else named
        return _scored(chosen, 0.82, signature)
    if named:
        return _scored(named, 0.78, evidence)
    if signature:
        return _scored(signature, 0.6, signature)
    return Scored()


def _description(lines: list[str], text: str) -> Scored[str]:
    for index, line in enumerate(lines):
        match = _DESC_LABEL.match(line.strip())
        if not match:
            continue
        parts = [match.group(1).strip()] if match.group(1).strip() else []
        for follow in lines[index + 1 :]:
            stripped = follow.strip()
            if not stripped:
                break
            if (
                _STOP_LABEL.match(stripped)
                or _DESC_LABEL.match(stripped)
                or classify_money_line(stripped)
                or _parse_item_line(stripped)
                or _is_header(stripped)
            ):
                break
            parts.append(stripped)
        paragraph = norm_ws(" ".join(part for part in parts if part))
        if paragraph:
            return _scored(paragraph, 0.88, paragraph)
    flat = norm_ws(text)
    sentences = re.split(r"(?<=[.!?])\s+", flat)
    candidates = [
        sentence.strip()
        for sentence in sentences
        if _WORK.search(sentence) and len(find_money(sentence)) < 2
    ]
    if candidates:
        best = max(candidates, key=lambda sentence: len(sentence.split()))
        if len(best.split()) >= 6:
            return _scored(best, 0.66, best)
    return Scored()


def _schedule(text: str) -> Scored[int]:
    for pattern, confidence in _SCHEDULE:
        match = pattern.search(text)
        if not match:
            continue
        if match.lastindex is None:
            return _scored(0, confidence, match.group(0))
        raw = match.group(1)
        if raw.casefold() in _WORD_NUM:
            return _scored(_WORD_NUM[raw.casefold()], confidence, match.group(0))
        return _scored(int(raw), confidence, match.group(0))
    return Scored()


def _reason(text: str) -> Scored[str]:
    for pattern in _REASON:
        match = pattern.search(text)
        if not match:
            continue
        value = match.group(1).strip(" .")
        if value:
            return _scored(value, 0.84, match.group(0))
    fallback = re.search(r"\b(unforeseen(?:\s+existing\s+condition)?)\b", text, re.I)
    if fallback:
        return _scored(fallback.group(1), 0.7, fallback.group(0))
    return Scored()


def _approval(text: str) -> Scored[str]:
    lines = [
        line.strip()
        for line in text.splitlines()
        if re.search(r"\b(status|approval|signatures?)\b", line, re.I)
    ]
    if lines:
        blob = " ".join(lines)
        status = _classify_approval(blob)
        if status is not None:
            return _scored(status, 0.86, blob)
        return Scored()
    low = text.casefold()
    if "not signed" in low or ("still need" in low and "signature" in low):
        sentence = _sentence_with(text, "signed") or _sentence_with(text, "signature")
        return _scored("pending", 0.68, sentence)
    return Scored()


def _classify_approval(blob: str) -> str | None:
    low = blob.casefold()
    if "draft" in low:
        return "draft"
    if "reject" in low or "denied" in low:
        return "rejected"
    if any(
        token in low
        for token in ("pending", "not signed", "blank", "____", "waiting", "still need")
    ):
        return "pending"
    if "approved" in low or re.search(r"\bsigned\b", low):
        return "approved"
    return None


def _currency(text: str) -> Scored[str]:
    if "$" in text:
        return _scored("USD", 0.9, "$")
    return Scored()


def _company_line(lines: list[str]) -> str | None:
    for line in reversed(lines):
        stripped = line.strip().lstrip("-").strip()
        if not stripped or "@" in stripped or len(stripped) > 80 or len(stripped.split()) > 8:
            continue
        if re.match(r"(?i)^(from|to|subject|project|owner|date|status)\b", stripped):
            continue
        if _COMPANY.search(stripped):
            return stripped
    return None


def _item(
    description: str,
    quantity: Decimal,
    unit: str,
    price: Decimal,
    amount: Decimal,
    evidence: str,
    confidence: float,
) -> LineItem:
    return LineItem(
        description=_scored(description, confidence, evidence),
        quantity=_scored(quantity, confidence, evidence),
        unit=_scored(unit, confidence, evidence),
        unit_price=_scored(price, confidence, evidence),
        amount=_scored(amount, confidence, evidence),
    )


def _scored(value: object, confidence: float, evidence: str | None) -> Scored[object]:
    quote = norm_ws(evidence) if evidence else None
    return Scored(
        value=value,
        confidence=round(min(0.93, max(0.0, confidence)), 3),
        needs_review=False,
        evidence=quote,
    )


def _clean_desc(description: str) -> str:
    description = re.sub(r"[\s.]+$", "", description.strip())
    description = re.sub(r"\s*\.+\s*$", "", description)
    return norm_ws(description).strip(" -")


def _trim_leading(description: str) -> str:
    trimmed = _LEADING.sub("", description).strip()
    return trimmed or description


def _clean_party(value: str) -> str:
    value = norm_ws(value)
    value = re.sub(r"\s*\((?:no\.?|job|project)\s*[^)]*\)\s*$", "", value, flags=re.I)
    return value.strip(" -")


def _canonical_unit(unit: str) -> str:
    token = norm_ws(unit).casefold()
    aliases = {
        "square feet": "SF",
        "square foot": "SF",
        "sq ft": "SF",
        "sq. ft": "SF",
        "sq. ft.": "SF",
        "linear feet": "LF",
        "linear foot": "LF",
        "each": "EA",
        "hour": "HR",
        "hours": "HR",
        "lump sum": "LS",
        "days": "DAY",
        "day": "DAY",
    }
    return aliases.get(token, token.upper())


def _sum_amounts(items: list[LineItem]) -> Decimal | None:
    amounts = [item.amount.value for item in items if item.amount.value is not None]
    if not amounts:
        return None
    return quantize_money(sum(amounts, Decimal("0")))


def _shares_token(left: str, right: str) -> bool:
    left_tokens = {token for token in re.findall(r"[A-Za-z]{4,}", left.casefold())}
    right_tokens = {token for token in re.findall(r"[A-Za-z]{4,}", right.casefold())}
    return bool(left_tokens & right_tokens)


def _sentence_with(text: str, needle: str) -> str | None:
    flat = norm_ws(text)
    for sentence in re.split(r"(?<=[.!?])\s+", flat):
        if needle.casefold() in sentence.casefold():
            return sentence
    return None
