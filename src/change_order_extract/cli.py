"""Command line interface.

Examples
--------
change-order extract samples/co_01_aia_clean.txt
change-order extract scan.pdf -o out.json --provider rules
change-order eval --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from change_order_extract.evaluate import default_samples_dir, evaluate_dir, format_misses, format_report
from change_order_extract.pipeline import extract_path, extract_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="change-order")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="extract one PDF or text file")
    extract.add_argument("input", help="path to a .pdf, .txt, or .md file, or - for stdin")
    extract.add_argument("-o", "--output", help="write JSON here instead of stdout")
    extract.add_argument("--provider", choices=("auto", "rules", "openai"), default="auto")

    evaluate = sub.add_parser("eval", help="score the labeled text fixtures")
    evaluate.add_argument("--samples", type=Path, default=None, help="fixtures directory")
    evaluate.add_argument("--provider", choices=("auto", "rules", "openai"), default="rules")
    evaluate.add_argument("--json", action="store_true", help="print the report as JSON")
    evaluate.add_argument("--verbose", action="store_true", help="list misses and false positives")

    args = parser.parse_args(argv)
    try:
        if args.command == "extract":
            return _extract(args.input, args.output, args.provider)
        return _eval(args.samples, args.provider, as_json=args.json, verbose=args.verbose)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _extract(source: str, output: str | None, provider: str) -> int:
    if source == "-":
        result = extract_text(sys.stdin.read(), provider=provider, source_name="stdin")
    else:
        result = extract_path(source, provider=provider)
    payload = result.model_dump_json(indent=2)
    if output:
        Path(output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


def _eval(samples: Path | None, provider: str, *, as_json: bool, verbose: bool) -> int:
    report = evaluate_dir(samples or default_samples_dir(), provider=provider)
    if as_json:
        payload = {
            "documents": len(report.documents),
            "labeled_correct": sum(row.correct for row in report.labeled),
            "labeled_total": len(report.labeled),
            "labeled_accuracy": report.labeled_accuracy(),
            "false_positives": len(report.false_positive_rows),
            "per_document": [
                {
                    "id": document.doc_id,
                    "labeled_correct": document.labeled_correct,
                    "labeled_total": document.labeled_total,
                    "document_confidence": document.document_confidence,
                    "review_flags": document.review_flags,
                }
                for document in report.documents
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(format_report(report))
    if verbose:
        misses = format_misses(report)
        if misses:
            print()
            print(misses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
