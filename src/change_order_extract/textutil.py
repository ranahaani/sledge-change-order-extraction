"""Normalization, parsing, and source-grounding helpers.

OCR repair is deliberately narrow. ``0`` glued to letters becomes ``O``.
A ``1`` glued to letters is rewritten only when exactly one lexicon word
matches. Real digit errors (an 8 read as a 9) are left alone.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from itertools import product

# Words the OCR repair is allowed to restore. Tokens that are not in this
# list are not guessed. Kept in-repo so results do not depend on the host
# dictionary.
_LEXICON_WORDS = """
amount april approved asphalt authority backcharge bead behind boiler bond
builders calendar cast change civil clinic coating concrete condensate
connection contractor contractors contract cracked crane credit curtain
damage date days delete description development differing district draft
drywall each electric epoxy exempt existing extend feet fittings follow
furnish galvanized glazing grand gypsum harbor haul health hours impact
inch injection inspection install insulation interior interiors iron
kenmore labor laminate latent library lift line linear lump markup
mechanical medical millwork mobilization municipal night number order
overhead owner owners paint parking patch pending percent pinnacle pipe
piping plastic plates premium price profit project protection pump quantity
reason rebar relocation replace residence restoration restraints return
revision roofing sales schedule school shield signature signed site slab
state station steel subtotal substitute surface tax total tower trench
unforeseen unit upgrade utility value wall water welding west working
addition amount approved blank change client code condition days design
engineering exit grand net pending request revision safety signs
"""

LEXICON = frozenset(_LEXICON_WORDS.split())

_MONEY = re.compile(
    r"""
    (?P<token>
        \(\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})\)
        |-\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})
        |\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?
        |(?<![\d.])\d{1,3}(?:,\d{3})*\.\d{2}(?!\d)
    )
    """,
    re.VERBOSE,
)

_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

_WORD_DAY = {"one": 1, "two": 2, "three": 3}

UNIT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "SF": ("sf", "sq ft", "sq. ft", "square feet", "square foot"),
    "LF": ("lf", "linear feet", "linear foot", "lin ft"),
    "EA": ("ea", "each"),
    "LS": ("ls", "lump sum"),
    "HR": ("hr", "hour", "hours"),
    "SY": ("sy", "square yard", "square yards"),
    "LB": ("lb", "pound", "pounds"),
    "DAY": ("day", "days"),
    "CY": ("cy", "cubic yard", "cubic yards"),
}


def norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def norm_text(text: str) -> str:
    """Casefold and strip punctuation for comparison."""

    text = text.casefold().replace("&", " and ")
    text = re.sub(r"[^a-z0-9.\s-]", " ", text)
    return norm_ws(text)


def token_f1(left: str, right: str) -> float:
    a = set(norm_text(left).split())
    b = set(norm_text(right).split())
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    if overlap == 0:
        return 0.0
    precision = overlap / len(a)
    recall = overlap / len(b)
    return 2 * precision * recall / (precision + recall)


def strings_match(predicted: str, expected: str, *, min_f1: float = 0.67) -> bool:
    if norm_text(predicted) == norm_text(expected):
        return True
    return token_f1(predicted, expected) >= min_f1


def quantize_money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"))


def parse_decimal(token: str) -> Decimal | None:
    raw = token.strip()
    if not raw:
        return None
    negative = raw.startswith("(") or raw.startswith("-")
    cleaned = raw.replace("$", "").replace(",", "").replace("(", "").replace(")", "")
    cleaned = cleaned.replace("-", "").strip()
    if not re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
        return None
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    return -value if negative else value


def find_money(line: str) -> list[Decimal]:
    """Money tokens on a line. Numbers that are percentages are skipped."""

    found: list[Decimal] = []
    for match in _MONEY.finditer(line):
        if line[match.end() : match.end() + 1] == "%":
            continue
        value = parse_decimal(match.group("token"))
        if value is not None:
            found.append(quantize_money(value))
    return found


def find_percent(line: str) -> Decimal | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
    if not match:
        return None
    return Decimal(match.group(1))


def parse_date(text: str) -> date | None:
    iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if iso:
        return _safe_date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
    us = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", text)
    if us:
        month, day, year = int(us.group(1)), int(us.group(2)), int(us.group(3))
        if year < 100:
            year += 2000
        return _safe_date(year, month, day)
    named = re.search(r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b", text)
    if named:
        month = _MONTHS.get(named.group(1).casefold())
        if month is not None:
            return _safe_date(int(named.group(3)), month, int(named.group(2)))
    return None


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def schedule_mentions(text: str) -> set[int]:
    found: set[int] = set()
    for match in re.finditer(r"(-?\d+)\s+(?:calendar\s+|working\s+)?days?\b", text, re.I):
        found.add(int(match.group(1)))
    for word, number in _WORD_DAY.items():
        if re.search(rf"\b{word}\s+days?\b", text, re.I):
            found.add(number)
    if re.search(r"\bno schedule (?:hit|impact|change)\b", text, re.I):
        found.add(0)
    return found


def normalize_ocr_text(text: str) -> tuple[str, int]:
    """Repair a few OCR confusions. Returns the text and a change count."""

    cleaned = (
        text.replace("\u00a0", " ")
        .replace("\u2014", " - ")
        .replace("\u2013", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    parts = re.split(r"(\s+)", cleaned)
    changes = 0
    output: list[str] = []
    for part in parts:
        if not part or part.isspace():
            output.append(part)
            continue
        fixed = _normalize_token(part)
        if fixed != part:
            changes += 1
        output.append(fixed)
    return "".join(output), changes


def _normalize_token(token: str) -> str:
    match = re.match(r"^(\W*)(.*?)(\W*)$", token, re.DOTALL)
    if not match:
        return token
    prefix, body, suffix = match.groups()
    if not body or not re.search(r"[01]", body):
        return token
    pieces = re.split(r"(-)", body)
    fixed = [_normalize_piece(piece) if piece != "-" else piece for piece in pieces]
    return prefix + "".join(fixed) + suffix


def _normalize_piece(piece: str) -> str:
    if not piece or not re.search(r"[01]", piece):
        return piece
    letters = sum(character.isalpha() for character in piece)
    digits = sum(character.isdigit() for character in piece)
    if digits > letters:
        return piece.translate(str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1"}))
    if letters == 0:
        return piece
    piece = _fix_zeros(piece)
    return _fix_ones(piece)


def _adjacent_letter(text: str, index: int) -> bool:
    previous = index > 0 and text[index - 1].isalpha()
    following = index + 1 < len(text) and text[index + 1].isalpha()
    return previous or following


def _fix_zeros(piece: str) -> str:
    chars = list(piece)
    for index, character in enumerate(chars):
        if character != "0" or not _adjacent_letter(piece, index):
            continue
        neighbor = chars[index - 1] if index > 0 and chars[index - 1].isalpha() else chars[index + 1]
        chars[index] = "O" if neighbor.isupper() else "o"
    return "".join(chars)


def _fix_ones(piece: str) -> str:
    if "11" in piece:
        candidate = piece.replace("11", "ll")
        if candidate.casefold() in LEXICON:
            piece = candidate
    positions = [
        index
        for index, character in enumerate(piece)
        if character == "1" and _adjacent_letter(piece, index)
    ]
    if not positions or len(piece) < 3:
        return piece
    choices = ("I", "L") if piece.isupper() else ("i", "l")
    matches: list[str] = []
    for picks in product(choices, repeat=len(positions)):
        chars = list(piece)
        for index, pick in zip(positions, picks, strict=True):
            chars[index] = pick
        candidate = "".join(chars)
        if candidate.casefold() in LEXICON:
            matches.append(candidate)
    unique = list(dict.fromkeys(matches))
    if len(unique) == 1:
        return unique[0]
    return piece


def grounded_money(value: Decimal, source: str) -> bool:
    amount = quantize_money(abs(value))
    raw = source.replace(",", "")
    variants = {f"{amount:.2f}", f"{amount:,.2f}"}
    if amount == amount.to_integral():
        variants.add(str(int(amount)))
        variants.add(f"{int(amount):,}")
    return any(variant in source or variant in raw for variant in variants)


def grounded_date(value: date, source: str) -> bool:
    month = value.strftime("%B")
    short = value.strftime("%b")
    forms = [
        value.isoformat(),
        f"{value.month}/{value.day}/{value.year}",
        f"{value.month:02d}/{value.day:02d}/{value.year}",
        f"{value.month}/{value.day}/{value.year % 100:02d}",
        f"{month} {value.day}, {value.year}",
        f"{month} {value.day} {value.year}",
        f"{short} {value.day}, {value.year}",
        f"{short} {value.day} {value.year}",
    ]
    folded = source.casefold()
    return any(form.casefold() in folded for form in forms)


def grounded_identifier(value: str, source: str) -> bool:
    return (
        re.search(rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])", source, re.I)
        is not None
    )


def grounded_string(value: str, source: str) -> bool:
    folded_value = norm_ws(value.casefold())
    folded_source = norm_ws(source.casefold())
    if folded_value and folded_value in folded_source:
        return True
    tokens = [token for token in re.findall(r"[a-z0-9]+", folded_value) if len(token) > 2]
    if not tokens:
        return False
    hits = sum(1 for token in tokens if token in folded_source)
    return hits / len(tokens) >= 0.8


def quote_in_source(evidence: str | None, source: str) -> bool:
    if not evidence or len(norm_ws(evidence)) < 2:
        return False
    return norm_ws(evidence).casefold() in norm_ws(source).casefold()
