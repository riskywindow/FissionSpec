#!/usr/bin/env python3
"""Generate, audit, and verify hash-locked FissionSpec output corpora."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from fissionspec.artifacts import canonical_json_bytes
from fissionspec.output_audit import (
    CORPUS_SCHEMA,
    REPORT_SCHEMA,
    AuditThresholds,
    OutputAuditError,
    audit_corpus,
    generate_synthetic_fixture,
    load_corpus,
    load_report,
    write_document,
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise OutputAuditError(f"duplicate JSON object key: {key!r}")
        document[key] = value
    return document


def _strict_json(path: Path) -> dict[str, object]:
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON numeric constant: {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OutputAuditError(f"cannot read strict JSON: {path}") from error
    if not isinstance(document, dict):
        raise OutputAuditError(f"JSON root must be an object: {path}")
    return cast(dict[str, object], document)


def _thresholds(path: Path | None) -> AuditThresholds:
    return AuditThresholds() if path is None else AuditThresholds.from_mapping(_strict_json(path))


def _print_summary(
    *,
    corpus_path: Path,
    report_path: Path,
    report: dict[str, object],
) -> None:
    gate = cast(dict[str, object], report["gate"])
    aggregate = cast(dict[str, object], report["aggregate"])
    summary = {
        "corpus": str(corpus_path),
        "corpus_payload_sha256": report["corpus_payload_sha256"],
        "report": str(report_path),
        "report_payload_sha256": report["payload_sha256"],
        "status": gate["status"],
        "violations": gate["violation_count"],
        "records": aggregate["records"],
        "clusters": aggregate["clusters"],
        "measurement_warning": report["measurement_warning"],
    }
    print(canonical_json_bytes(summary).decode(), end="")


def _fixture(args: argparse.Namespace) -> int:
    output = Path(args.output_directory)
    corpus_path = output / "synthetic_output_corpus.json"
    report_path = output / "synthetic_output_audit_report.json"
    corpus = generate_synthetic_fixture(args.records)
    thresholds = _thresholds(args.thresholds)
    report = audit_corpus(corpus, thresholds=thresholds)
    write_document(corpus_path, corpus)
    write_document(report_path, report)
    _print_summary(corpus_path=corpus_path, report_path=report_path, report=report)
    gate = cast(dict[str, object], report["gate"])
    return 0 if gate["status"] == "pass" else 1


def _audit(args: argparse.Namespace) -> int:
    corpus_path = Path(args.corpus)
    report_path = Path(args.report)
    corpus = load_corpus(corpus_path)
    report = audit_corpus(corpus, thresholds=_thresholds(args.thresholds))
    write_document(report_path, report)
    _print_summary(corpus_path=corpus_path, report_path=report_path, report=report)
    gate = cast(dict[str, object], report["gate"])
    return 0 if gate["status"] == "pass" else 1


def _verify(args: argparse.Namespace) -> int:
    path = Path(args.document)
    raw = _strict_json(path)
    if raw.get("schema") == CORPUS_SCHEMA:
        document = load_corpus(path)
    elif raw.get("schema") == REPORT_SCHEMA:
        document = load_report(path)
    else:
        raise OutputAuditError("document has an unsupported or missing schema")
    print(
        canonical_json_bytes(
            {
                "document": str(path),
                "payload_sha256": document["payload_sha256"],
                "schema": document["schema"],
                "status": "verified",
            }
        ).decode(),
        end="",
    )
    return 0


def _write_thresholds(args: argparse.Namespace) -> int:
    destination = Path(args.path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(AuditThresholds().as_dict()))
    temporary.replace(destination)
    print(f"wrote preregistration template: {destination}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic CPU analysis over paired output distributions captured once "
            "from a serving engine."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fixture = subparsers.add_parser(
        "fixture",
        help="generate and audit a deterministic synthetic CPU fixture",
    )
    fixture.add_argument("output_directory")
    fixture.add_argument("--records", type=int, default=512)
    fixture.add_argument("--thresholds", type=Path)
    fixture.set_defaults(handler=_fixture)

    audit = subparsers.add_parser("audit", help="audit an existing verified corpus")
    audit.add_argument("corpus")
    audit.add_argument("report")
    audit.add_argument("--thresholds", type=Path)
    audit.set_defaults(handler=_audit)

    verify = subparsers.add_parser("verify", help="verify a corpus or report hash envelope")
    verify.add_argument("document")
    verify.set_defaults(handler=_verify)

    thresholds = subparsers.add_parser(
        "write-default-thresholds",
        help="write the complete strict preregistration mapping",
    )
    thresholds.add_argument("path")
    thresholds.set_defaults(handler=_write_thresholds)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested offline output-audit command."""

    args = _parser().parse_args(argv)
    try:
        return cast(int, args.handler(args))
    except (OutputAuditError, OSError, ValueError) as error:
        print(f"output audit failed closed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
