#!/usr/bin/env python3
"""Validate the FissionSpec patch series against its exact SPECTRE PR head."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

INTEGRATION_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = INTEGRATION_DIR / "patch_manifest.json"

CHANGED_PYTHON = (
    "python/sglang/srt/managers/scheduler.py",
    "python/sglang/srt/speculative/spectre/fission_state.py",
    "python/sglang/srt/speculative/spectre/spectre_protocol.py",
    "python/sglang/srt/speculative/spectre/drafter/spectre_state_manager.py",
    "python/sglang/srt/speculative/spectre/drafter/spectre_draft_scheduler_mixin.py",
    "python/sglang/srt/speculative/spectre/verifier/spectre_worker.py",
    "python/sglang/srt/speculative/spectre/verifier/spectre_target_scheduler_mixin.py",
)
RUFF_PYTHON = tuple(
    source for source in CHANGED_PYTHON if source != "python/sglang/srt/managers/scheduler.py"
)
CPU_TESTS = (
    "test/srt/test_spectre_fission_state.py",
    "test/srt/test_spectre_fission_scheduler.py",
    "test/srt/test_spectre_fission_protocol.py",
)


class ValidationError(RuntimeError):
    pass


def _run(
    args: list[str],
    *,
    cwd: Path,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        text=capture,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if result.returncode != 0:
        detail = ""
        if capture:
            detail = f"\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        raise ValidationError(f"command failed ({result.returncode}): {' '.join(args)}{detail}")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(document: object) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_git_blob(repository: Path, commit: str, source_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{source_path}"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValidationError(
            f"cannot read {source_path} at {commit}: {result.stderr.decode(errors='replace')}"
        )
    return hashlib.sha256(result.stdout).hexdigest()


def _load_manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    expected_keys = {
        "schema",
        "generated_at",
        "repository",
        "target",
        "current_main_at_audit",
        "series",
        "source_preimages",
        "cpu_validation",
        "payload_sha256",
        "runtime_boundary",
    }
    if set(manifest) != expected_keys:
        raise ValidationError("patch manifest has unexpected or missing fields")
    if manifest.get("schema") != "fissionspec.sglang-patch-series.v1":
        raise ValidationError("unexpected patch manifest schema")
    supplied_hash = manifest.get("payload_sha256")
    if not isinstance(supplied_hash, str) or re.fullmatch(r"[0-9a-f]{64}", supplied_hash) is None:
        raise ValidationError("invalid patch manifest payload hash")
    payload = dict(manifest)
    payload.pop("payload_sha256")
    actual_hash = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if actual_hash != supplied_hash:
        raise ValidationError(
            f"patch manifest payload hash mismatch: {actual_hash} != {supplied_hash}"
        )
    return manifest


def _verify_artifact_hashes(manifest: dict, upstream: Path) -> list[Path]:
    patch_paths = []
    expected_orders = list(range(1, len(manifest["series"]) + 1))
    actual_orders = [item["order"] for item in manifest["series"]]
    if actual_orders != expected_orders:
        raise ValidationError("patch series order is not contiguous")

    for item in manifest["series"]:
        if set(item) != {"order", "path", "sha256", "commit", "subject"}:
            raise ValidationError("patch series entry has unexpected or missing fields")
        if (
            not isinstance(item["commit"], str)
            or re.fullmatch(r"[0-9a-f]{40}", item["commit"]) is None
        ):
            raise ValidationError("patch series commit is not a full object ID")
        if (
            not isinstance(item["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
        ):
            raise ValidationError("patch series checksum is malformed")
        patch_path = INTEGRATION_DIR / item["path"]
        if not patch_path.is_file():
            raise ValidationError(f"missing patch: {patch_path}")
        actual = _sha256_file(patch_path)
        if actual != item["sha256"]:
            raise ValidationError(
                f"patch checksum mismatch for {patch_path.name}: {actual} != {item['sha256']}"
            )
        patch_paths.append(patch_path)

    head = manifest["target"]["head_commit"]
    _run(["git", "cat-file", "-e", f"{head}^{{commit}}"], cwd=upstream)
    for source in manifest["source_preimages"]:
        actual = _sha256_git_blob(upstream, head, source["path"])
        if actual != source["sha256"]:
            raise ValidationError(
                f"source preimage mismatch for {source['path']}: {actual} != {source['sha256']}"
            )
    return patch_paths


def _static_contract_checks(worktree: Path) -> None:
    state = (worktree / "python/sglang/srt/speculative/spectre/fission_state.py").read_text(
        encoding="utf-8"
    )
    worker = (
        worktree / "python/sglang/srt/speculative/spectre/verifier/spectre_worker.py"
    ).read_text(encoding="utf-8")
    target = (
        worktree
        / ("python/sglang/srt/speculative/spectre/verifier/spectre_target_scheduler_mixin.py")
    ).read_text(encoding="utf-8")
    protocol = (worktree / "python/sglang/srt/speculative/spectre/spectre_protocol.py").read_text(
        encoding="utf-8"
    )
    drafter = (
        worktree
        / ("python/sglang/srt/speculative/spectre/drafter/spectre_draft_scheduler_mixin.py")
    ).read_text(encoding="utf-8")
    scheduler = (worktree / "python/sglang/srt/managers/scheduler.py").read_text(encoding="utf-8")

    required_state = (
        "class FissionKey",
        "fission_version: int",
        "class FissionOutcomeEvent",
        "class FissionStateTable",
        "def fission_wire_key(",
        "def fission_control_matches(",
        'source.get("SGLANG_SPECTRE_FISSION", "0")',
    )
    required_target = (
        "batch.filter_batch(",
        "self._fission_parked",
        "self._send_retry_requests(",
        "self._fission_state.complete_recovery(",
        "real_verifier_slots",
        "graph_bucket_slots",
        "recovery_ages_ms",
        "require_version=True",
    )
    required_drafter = (
        "fission_identity_required",
        "fission_control_matches(",
        "malformed draft identity reached request creation",
    )
    for marker in required_state:
        if marker not in state:
            raise ValidationError(f"missing state contract marker: {marker}")
    for marker in required_target:
        if marker not in target:
            raise ValidationError(f"missing target contract marker: {marker}")
    for marker in required_drafter:
        if marker not in drafter:
            raise ValidationError(f"missing drafter contract marker: {marker}")
    if "fission_version: Optional[int] = None" not in protocol:
        raise ValidationError("wire version is not rolling-compatible")
    if "if should_retry and not fission_enabled:" not in worker:
        raise ValidationError("fission path is not fenced from synchronous retry")
    if "self._fission_cancel_parked(req)" not in scheduler:
        raise ValidationError("parked cancellation is not wired into Scheduler")


def _validate_in_worktree(
    upstream: Path,
    head: str,
    patch_paths: list[Path],
    *,
    skip_ruff: bool,
) -> None:
    temporary_root = Path(tempfile.mkdtemp(prefix="fissionspec-sglang-validate.")).resolve()
    worktree = temporary_root / "sglang"
    added = False
    try:
        _run(
            ["git", "worktree", "add", "--detach", str(worktree), head],
            cwd=upstream,
        )
        added = True
        actual_head = _run(["git", "rev-parse", "HEAD"], cwd=worktree, capture=True).stdout.strip()
        if actual_head != head:
            raise ValidationError(f"worktree head drift: {actual_head} != {head}")

        for patch_path in patch_paths:
            # The explicit check is the acceptance artifact; patches after the
            # first are checked against the prior patches in series order.
            _run(["git", "apply", "--check", str(patch_path)], cwd=worktree)
            _run(["git", "apply", str(patch_path)], cwd=worktree)

        _run(["git", "diff", "--check"], cwd=worktree)
        _static_contract_checks(worktree)

        if not skip_ruff:
            if shutil.which("ruff") is None:
                raise ValidationError("ruff is required (or pass --skip-ruff)")
            _run(
                ["ruff", "check", *RUFF_PYTHON, *CPU_TESTS],
                cwd=worktree,
            )

        _run([sys.executable, "-m", "py_compile", *CHANGED_PYTHON], cwd=worktree)
        for test_path in CPU_TESTS:
            _run([sys.executable, test_path], cwd=worktree)
    finally:
        if added:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=upstream,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        shutil.rmtree(temporary_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sglang",
        required=True,
        type=Path,
        help="local SGLang git repository containing the pinned PR head",
    )
    parser.add_argument(
        "--skip-ruff",
        action="store_true",
        help="skip Ruff only; syntax, artifact, apply, and CPU tests still run",
    )
    args = parser.parse_args()
    upstream = args.sglang.resolve()
    if not (upstream / ".git").exists() and not (
        _run(
            ["git", "rev-parse", "--git-dir"],
            cwd=upstream,
            capture=True,
        ).stdout
    ):
        raise ValidationError(f"not a git repository: {upstream}")

    manifest = _load_manifest()
    patch_paths = _verify_artifact_hashes(manifest, upstream)
    _validate_in_worktree(
        upstream,
        manifest["target"]["head_commit"],
        patch_paths,
        skip_ruff=args.skip_ruff,
    )
    print(
        f"PASS: {len(manifest['series'])} patches apply in order to "
        f"{manifest['target']['head_commit']}; "
        f"{manifest['cpu_validation']['tests']} CPU tests declared"
    )
    print("NOT TESTED: target kernels and backend-specific KV/descriptor reinsertion correctness")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from None
