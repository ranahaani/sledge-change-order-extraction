"""Score extractions against the labeled fixtures.

Gold files store the values a careful reader would copy off the page,
including a stated total that does not match the line items. Accuracy is
"did we read what was written", not "did we repair the document".

Labeled accuracy ignores fields that are null on both sides. A value
predicted for a null gold field is a false positive. Calibration uses
labeled fields plus those false positives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from change_order_extract.pipeline import extract_text
from change_order_extract.schema import ChangeOrderExtraction, LineItem, Scored
from change_order_extract.textutil import strings_match, token_f1

HIGH_CONFIDENCE = 0.8
_BINS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0001))

_TEXT_FIELDS = {
    "project_name": 0.67,
    "contractor": 0.67,
    "owner": 0.67,
    "description": 0.5,
    "reason": 0.5,
}
_ID_FIELDS = ("project_number", "change_order_number", "currency", "approval_status")
_MONEY_FIELDS = ("line_subtotal", "markup", "tax", "other_fees", "total")
_PERCENT_FIELDS = ("markup_percent", "tax_percent")


@dataclass
class FieldScore:
    doc_id: str
    field: str
    correct: bool
    confidence: float
    needs_review: bool
    predicted: str | None
    expected: str | None
    gold_present: bool
    false_positive: bool


@dataclass
class DocumentSummary:
    doc_id: str
    labeled_correct: int
    labeled_total: int
    document_confidence: float
    review_flags: list[str]


@dataclass
class EvalReport:
    rows: list[FieldScore] = field(default_factory=list)
    documents: list[DocumentSummary] = field(default_factory=list)

    @property
    def labeled(self) -> list[FieldScore]:
        return [row for row in self.rows if row.gold_present]

    @property
    def false_positive_rows(self) -> list[FieldScore]:
        return [row for row in self.rows if row.false_positive]

    def labeled_accuracy(self) -> float:
        rows = self.labeled
        if not rows:
            return 0.0
        return sum(row.correct for row in rows) / len(rows)

    def high_confidence_accuracy(self) -> tuple[float, int, int]:
        rows = [row for row in self.labeled if row.confidence >= HIGH_CONFIDENCE]
        if not rows:
            return 0.0, 0, 0
        correct = sum(row.correct for row in rows)
        return correct / len(rows), correct, len(rows)

    def calibration_pairs(self) -> list[tuple[float, bool]]:
        return [
            (row.confidence, row.correct)
            for row in self.rows
            if row.gold_present or row.false_positive
        ]


def default_samples_dir() -> Path:
    candidate = Path(__file__).resolve().parents[2] / "samples"
    if candidate.exists():
        return candidate
    return Path("samples")


def discover(samples_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for text_path in sorted(samples_dir.glob("*.txt")):
        gold = samples_dir / "expected" / f"{text_path.stem}.json"
        if gold.is_file():
            pairs.append((text_path, gold))
    return pairs


def evaluate_dir(samples_dir: Path | None = None, *, provider: str = "rules") -> EvalReport:
    root = samples_dir or default_samples_dir()
    report = EvalReport()
    for text_path, gold_path in discover(root):
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        prediction = extract_text(
            text_path.read_text(encoding="utf-8"),
            provider=provider,
            source_name=text_path.name,
        )
        rows = score_document(text_path.stem, prediction, gold)
        report.rows.extend(rows)
        labeled = [row for row in rows if row.gold_present]
        report.documents.append(
            DocumentSummary(
                doc_id=text_path.stem,
                labeled_correct=sum(row.correct for row in labeled),
                labeled_total=len(labeled),
                document_confidence=prediction.document_confidence,
                review_flags=list(prediction.review_flags),
            )
        )
    return report


def score_document(
    doc_id: str, prediction: ChangeOrderExtraction, gold: dict[str, object]
) -> list[FieldScore]:
    rows: list[FieldScore] = []
    for name, minimum in _TEXT_FIELDS.items():
        rows.append(_score_scalar(doc_id, name, getattr(prediction, name), gold.get(name), "text", minimum))
    for name in _ID_FIELDS:
        rows.append(_score_scalar(doc_id, name, getattr(prediction, name), gold.get(name), "id"))
    rows.append(_score_scalar(doc_id, "date", prediction.date, gold.get("date"), "date"))
    for name in _MONEY_FIELDS:
        rows.append(_score_scalar(doc_id, name, getattr(prediction, name), gold.get(name), "money"))
    for name in _PERCENT_FIELDS:
        rows.append(_score_scalar(doc_id, name, getattr(prediction, name), gold.get(name), "percent"))
    rows.append(
        _score_scalar(
            doc_id,
            "schedule_impact_days",
            prediction.schedule_impact_days,
            gold.get("schedule_impact_days"),
            "int",
        )
    )
    rows.extend(_score_lines(doc_id, prediction.line_items, gold.get("line_items") or []))
    return rows


def expected_calibration_error(pairs: list[tuple[float, bool]]) -> float:
    if not pairs:
        return 0.0
    total = len(pairs)
    error = 0.0
    for low, high in _BINS:
        chunk = [(confidence, ok) for confidence, ok in pairs if low <= confidence < high]
        if not chunk:
            continue
        accuracy = sum(ok for _confidence, ok in chunk) / len(chunk)
        mean_confidence = sum(confidence for confidence, _ok in chunk) / len(chunk)
        error += (len(chunk) / total) * abs(accuracy - mean_confidence)
    return error


def format_report(report: EvalReport) -> str:
    labeled = report.labeled
    labeled_correct = sum(row.correct for row in labeled)
    _ratio, high_correct, high_total = report.high_confidence_accuracy()
    pairs = report.calibration_pairs()
    correct_pairs = [confidence for confidence, ok in pairs if ok]
    wrong_pairs = [confidence for confidence, ok in pairs if not ok]
    errors = [row for row in labeled if not row.correct]
    flagged_errors = [row for row in errors if row.needs_review]
    flagged_correct = [row for row in labeled if row.correct and row.needs_review]
    lines = [
        "Change-order extraction eval",
        f"documents: {len(report.documents)}",
        f"labeled field accuracy: {_pct(labeled_correct, len(labeled))} ({labeled_correct}/{len(labeled)})",
        (
            "high-confidence accuracy (labeled, confidence >= 0.80): "
            f"{_pct(high_correct, high_total)} ({high_correct}/{high_total})"
        ),
        f"false positives: {len(report.false_positive_rows)}",
        f"mean confidence when correct: {_mean(correct_pairs)}",
        f"mean confidence when incorrect: {_mean(wrong_pairs)}",
        f"ECE: {expected_calibration_error(pairs):.3f}",
        (
            "value errors flagged for review: "
            f"{len(flagged_errors)}/{len(errors) if errors else 0}"
        ),
        f"correct values still flagged: {len(flagged_correct)}",
        "",
        "per field (labeled correct/total):",
    ]
    for name, correct, total in _grouped(labeled):
        lines.append(f"  {name:<28} {correct}/{total}")
    lines.append("")
    lines.append("per document (labeled correct/total, document confidence):")
    for document in report.documents:
        lines.append(
            f"  {document.doc_id:<32} {document.labeled_correct}/{document.labeled_total}"
            f"  conf {document.document_confidence:.3f}"
        )
    return "\n".join(lines)


def format_misses(report: EvalReport) -> str:
    lines: list[str] = []
    for row in report.rows:
        if row.gold_present and not row.correct:
            lines.append(
                f"MISS {row.doc_id} {row.field}: expected {row.expected!r} "
                f"got {row.predicted!r} conf {row.confidence:.3f} review {row.needs_review}"
            )
        elif row.false_positive:
            lines.append(
                f"FP   {row.doc_id} {row.field}: got {row.predicted!r} "
                f"conf {row.confidence:.3f} review {row.needs_review}"
            )
    return "\n".join(lines)


def _score_scalar(
    doc_id: str,
    name: str,
    field: Scored[object],
    expected: object,
    kind: str,
    min_f1: float = 0.67,
) -> FieldScore:
    predicted = field.value
    return FieldScore(
        doc_id=doc_id,
        field=name,
        correct=_values_match(predicted, expected, kind, min_f1=min_f1),
        confidence=field.confidence,
        needs_review=field.needs_review,
        predicted=_display(predicted),
        expected=_display(expected),
        gold_present=expected is not None,
        false_positive=expected is None and predicted is not None,
    )


def _score_lines(doc_id: str, predicted: list[LineItem], gold_items: object) -> list[FieldScore]:
    gold_list = gold_items if isinstance(gold_items, list) else []
    unused = set(range(len(predicted)))
    pairs: list[tuple[LineItem | None, dict[str, object]]] = []
    for gold in gold_list:
        if not isinstance(gold, dict):
            continue
        best_index = None
        best_score = 0.0
        for index in unused:
            score = token_f1(predicted[index].description.value or "", str(gold.get("description") or ""))
            if best_index is None or score > best_score:
                best_score = score
                best_index = index
        if best_index is not None and best_score >= 0.34:
            unused.remove(best_index)
            pairs.append((predicted[best_index], gold))
        else:
            pairs.append((None, gold))
    rows: list[FieldScore] = []
    for slot, (item, gold) in enumerate(pairs):
        rows.extend(_score_one_line(doc_id, slot, item, gold))
    for extra_index, index in enumerate(sorted(unused), start=len(pairs)):
        rows.extend(_score_one_line(doc_id, extra_index, predicted[index], {}))
    return rows


def _score_one_line(
    doc_id: str, slot: int, item: LineItem | None, gold: dict[str, object]
) -> list[FieldScore]:
    specs = (
        ("description", "text", 0.5),
        ("quantity", "qty", 0.0),
        ("unit", "unit", 0.0),
        ("unit_price", "money", 0.0),
        ("amount", "money", 0.0),
    )
    rows: list[FieldScore] = []
    for name, kind, minimum in specs:
        expected = gold.get(name) if gold else None
        if item is None:
            field = Scored()
        else:
            field = getattr(item, name)
        # An extra predicted line has an empty gold dict. Subfields that are
        # null on both sides are not false positives; a filled one is.
        if not gold and field.value is None:
            continue
        score = _score_scalar(doc_id, f"line_item.{name}", field, expected, kind, minimum)
        score.field = f"line_item[{slot}].{name}"
        rows.append(score)
    return rows


def _values_match(predicted: object, expected: object, kind: str, *, min_f1: float) -> bool:
    if expected is None:
        return predicted is None
    if predicted is None:
        return False
    if kind == "money":
        return abs(Decimal(str(predicted)) - Decimal(str(expected))) <= Decimal("0.01")
    if kind in {"percent", "qty"}:
        return abs(Decimal(str(predicted)) - Decimal(str(expected))) <= Decimal("0.001")
    if kind == "int":
        return int(str(predicted)) == int(str(expected))
    if kind == "date":
        return str(predicted) == str(expected)
    if kind in {"id", "unit"}:
        return str(predicted).casefold().replace(" ", "") == str(expected).casefold().replace(" ", "")
    return strings_match(str(predicted), str(expected), min_f1=min_f1)


def _display(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _grouped(rows: list[FieldScore]) -> list[tuple[str, int, int]]:
    order: list[str] = []
    buckets: dict[str, list[bool]] = {}
    for row in rows:
        name = row.field
        if name.startswith("line_item"):
            name = "line_item." + row.field.rsplit(".", 1)[1]
        if name not in buckets:
            order.append(name)
            buckets[name] = []
        buckets[name].append(row.correct)
    return [(name, sum(buckets[name]), len(buckets[name])) for name in order]


def _pct(correct: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{correct / total:.3f}"


def _mean(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"{sum(values) / len(values):.3f}"
