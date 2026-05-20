import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "patch_manifest.json"


def _canonical_json_bytes(document):
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


class TestPatchArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_manifest_and_series_are_pinned(self):
        self.assertEqual(
            set(self.manifest),
            {
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
            },
        )
        self.assertEqual(
            self.manifest["schema"],
            "fissionspec.sglang-patch-series.v1",
        )
        self.assertEqual(
            self.manifest["target"]["head_commit"],
            "1a8520879c53462b7ac1861d3aad7de4bf5860d4",
        )
        self.assertEqual(
            [item["order"] for item in self.manifest["series"]],
            [1, 2, 3, 4, 5],
        )
        payload = dict(self.manifest)
        supplied = payload.pop("payload_sha256")
        self.assertRegex(supplied, r"^[0-9a-f]{64}$")
        self.assertEqual(
            hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
            supplied,
        )

    def test_patch_hashes_and_mail_headers(self):
        total = len(self.manifest["series"])
        for index, item in enumerate(self.manifest["series"], start=1):
            self.assertEqual(
                set(item),
                {"order", "path", "sha256", "commit", "subject"},
            )
            self.assertRegex(item["commit"], r"^[0-9a-f]{40}$")
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")
            patch = ROOT / item["path"]
            payload = patch.read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])
            text = payload.decode("utf-8")
            self.assertTrue(text.startswith(f"From {item['commit']}"))
            self.assertIn(f"Subject: [PATCH {index}/{total}]", text)
            self.assertIn(item["subject"], text)
            self.assertNotIn("youremail@example.com", text)

    def test_first_patch_records_exact_base(self):
        first = ROOT / self.manifest["series"][0]["path"]
        self.assertIn(
            "base-commit: 1a8520879c53462b7ac1861d3aad7de4bf5860d4",
            first.read_text(encoding="utf-8"),
        )

    def test_wire_identity_hardening_is_in_the_series(self):
        final_patch = ROOT / self.manifest["series"][-1]["path"]
        text = final_patch.read_text(encoding="utf-8")
        for marker in (
            "def fission_wire_key(",
            "def fission_control_matches(",
            "fission_identity_required: bool = False",
            "require_version=True",
            "malformed draft identity reached request creation",
            "test_versioned_control_requires_an_exact_complete_key",
            "test_incomplete_or_malformed_fission_reply_cannot_fill_version_zero",
        ):
            self.assertIn(marker, text)
        self.assertEqual(self.manifest["cpu_validation"]["tests"], 23)

    def test_runtime_boundary_is_honest(self):
        boundary = self.manifest["runtime_boundary"]
        self.assertFalse(boundary["kernels_executed"])
        self.assertEqual(boundary["reinsertion_status"], "cpu_fake_only")
        documentation = (ROOT / "rebase_map.md").read_text(encoding="utf-8")
        self.assertIn("not a current-main patch", documentation)
        self.assertIn("GPU validation remains necessary", documentation)


if __name__ == "__main__":
    unittest.main()
