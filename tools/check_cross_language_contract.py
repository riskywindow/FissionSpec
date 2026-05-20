#!/usr/bin/env python3
"""Run both independent consumers of the reviewed parity fixtures."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, environment: dict[str, str] | None = None) -> int:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
    )
    return completed.returncode


def main() -> int:
    """Return zero only when Python and Rust accept the canonical corpus."""

    python_environment = dict(os.environ)
    python_environment["PYTHONPATH"] = str(ROOT / "src")
    python_status = _run(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_cross_language_contract",
            "-v",
        ],
        environment=python_environment,
    )
    if python_status:
        return python_status
    return _run(
        [
            "cargo",
            "test",
            "--manifest-path",
            "crates/fissionspec-core/Cargo.toml",
            "--test",
            "cross_language_contract",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
