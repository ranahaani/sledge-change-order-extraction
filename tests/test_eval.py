import pytest

from change_order_extract.evaluate import evaluate_dir, expected_calibration_error


def test_fixture_eval_matches_the_measured_baseline() -> None:
    """The README quotes these figures. Update both together."""

    report = evaluate_dir()
    labeled = report.labeled
    assert len(labeled) == 264
    assert sum(row.correct for row in labeled) == 264
    assert report.false_positive_rows == []
    _ratio, correct, total = report.high_confidence_accuracy()
    assert (correct, total) == (244, 244)
    flagged_correct = [row for row in labeled if row.correct and row.needs_review]
    assert len(flagged_correct) == 9
    assert expected_calibration_error(report.calibration_pairs()) == pytest.approx(0.10823106060606053)
    confidences = {document.doc_id: document.document_confidence for document in report.documents}
    assert confidences["co_01_aia_clean"] == 0.921
    assert confidences["co_03_inconsistent_total"] == 0.714
    assert confidences["co_06_margin_note"] == 0.813
    assert confidences["co_09_prose_items"] == 0.695
