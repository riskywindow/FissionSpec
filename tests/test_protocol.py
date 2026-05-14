"""Adversarial async ordering and recovery tests for the protocol FSM."""

from __future__ import annotations

import random
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from fissionspec.protocol import (
    FissionProtocol,
    InvalidMessageError,
    InvalidTransitionError,
    MessageTag,
    ProtocolSnapshotError,
    ProtocolState,
    RecoveryReply,
    RecoveryRequest,
    ReplyDisposition,
    VerifyReply,
)


class FissionProtocolTests(unittest.TestCase):
    def test_hit_path_and_next_round(self) -> None:
        protocol = FissionProtocol("request")
        request = protocol.start_verification()
        self.assertEqual(request.tag, MessageTag("request", 0, 1))
        self.assertEqual(protocol.state, ProtocolState.VERIFYING)

        transition = protocol.handle_verify_reply(
            VerifyReply(request.tag, hit=True, accepted_blocks=6)
        )
        self.assertTrue(transition.applied)
        self.assertEqual(transition.previous, ProtocolState.VERIFYING)
        self.assertEqual(transition.current, ProtocolState.READY_HIT)
        self.assertIsNone(transition.outbound)
        self.assertTrue(protocol.is_ready)

        next_request = protocol.start_verification()
        self.assertEqual(next_request.tag, MessageTag("request", 1, 2))
        self.assertEqual(protocol.state, ProtocolState.VERIFYING)

    def test_miss_recovery_to_backup_and_finished(self) -> None:
        protocol = FissionProtocol(17)
        verification = protocol.start_verification(round_id=4)
        miss = protocol.handle_verify_reply(
            VerifyReply(verification.tag, hit=False, accepted_blocks=2)
        )
        self.assertEqual(miss.current, ProtocolState.RECOVERING)
        self.assertEqual(miss.outbound, RecoveryRequest(verification.tag, 2))

        backup = protocol.handle_recovery_reply(
            RecoveryReply(verification.tag, finished=False, recovered_blocks=4)
        )
        self.assertTrue(backup.applied)
        self.assertEqual(protocol.state, ProtocolState.READY_BACKUP)

        next_verification = protocol.start_verification(round_id=9)
        protocol.handle_verify_reply(
            VerifyReply(next_verification.tag, hit=False, accepted_blocks=0)
        )
        finished = protocol.handle_recovery_reply(
            RecoveryReply(next_verification.tag, finished=True)
        )
        self.assertEqual(finished.current, ProtocolState.FINISHED)
        self.assertTrue(protocol.is_finished)
        self.assertFalse(protocol.is_ready)

    def test_wrong_request_round_version_and_duplicates_are_ignored(self) -> None:
        protocol = FissionProtocol("r")
        request = protocol.start_verification(round_id=7)
        stale_tags = [
            MessageTag("other", 7, 1),
            MessageTag("r", 6, 1),
            MessageTag("r", 7, 2),
        ]
        for tag in stale_tags:
            with self.subTest(tag=tag):
                result = protocol.handle_verify_reply(VerifyReply(tag, hit=True))
                self.assertEqual(result.disposition, ReplyDisposition.IGNORED_STALE)
                self.assertEqual(protocol.state, ProtocolState.VERIFYING)
                self.assertEqual(protocol.active_tag, request.tag)

        applied = protocol.handle_verify_reply(VerifyReply(request.tag, hit=True))
        self.assertTrue(applied.applied)
        duplicate = protocol.handle_verify_reply(VerifyReply(request.tag, hit=True))
        self.assertTrue(duplicate.ignored)
        self.assertEqual(protocol.state, ProtocolState.READY_HIT)

        stale_recovery = protocol.handle_recovery_reply(RecoveryReply(request.tag, finished=True))
        self.assertTrue(stale_recovery.ignored)
        self.assertEqual(protocol.state, ProtocolState.READY_HIT)

    def test_retry_fences_old_verifier_reply(self) -> None:
        protocol = FissionProtocol("r")
        old = protocol.start_verification(round_id=3)
        current = protocol.retry_verification()
        self.assertEqual(current.tag.round_id, old.tag.round_id)
        self.assertEqual(current.tag.version, old.tag.version + 1)

        ignored = protocol.handle_verify_reply(VerifyReply(old.tag, hit=True))
        self.assertTrue(ignored.ignored)
        self.assertEqual(protocol.active_tag, current.tag)
        applied = protocol.handle_verify_reply(VerifyReply(current.tag, hit=True))
        self.assertTrue(applied.applied)

    def test_recovery_fence_invalidates_both_old_reply_types(self) -> None:
        protocol = FissionProtocol("r")
        verify = protocol.start_verification()
        recovery = protocol.handle_verify_reply(
            VerifyReply(verify.tag, hit=False, accepted_blocks=3)
        ).outbound
        assert recovery is not None

        fenced = protocol.invalidate_inflight_for_recovery()
        self.assertEqual(fenced.accepted_blocks, 3)
        self.assertEqual(fenced.tag.version, recovery.tag.version + 1)
        self.assertTrue(protocol.handle_verify_reply(VerifyReply(verify.tag, hit=True)).ignored)
        self.assertTrue(protocol.handle_recovery_reply(RecoveryReply(recovery.tag)).ignored)
        self.assertEqual(protocol.state, ProtocolState.RECOVERING)
        self.assertTrue(protocol.handle_recovery_reply(RecoveryReply(fenced.tag)).applied)
        self.assertEqual(protocol.state, ProtocolState.READY_BACKUP)

    def test_exact_snapshot_restore_can_continue_inflight(self) -> None:
        protocol = FissionProtocol("r")
        request = protocol.start_verification(round_id=2)
        snapshot = protocol.snapshot()
        restored = FissionProtocol.from_snapshot(snapshot)
        self.assertEqual(restored.snapshot(), snapshot)
        result = restored.handle_verify_reply(VerifyReply(request.tag, hit=True))
        self.assertTrue(result.applied)
        self.assertEqual(restored.state, ProtocolState.READY_HIT)

        alias_restored = FissionProtocol.restore(restored.snapshot())
        self.assertEqual(alias_restored.state, ProtocolState.READY_HIT)

    def test_crash_resume_fences_precrash_inflight_reply(self) -> None:
        protocol = FissionProtocol("r")
        precrash = protocol.start_verification(round_id=5)
        resumed = FissionProtocol.resume_after_crash(protocol.recovery_snapshot())
        self.assertIsNotNone(resumed.outbound)
        assert resumed.outbound is not None
        self.assertEqual(resumed.protocol.state, ProtocolState.RECOVERING)
        self.assertEqual(resumed.outbound.tag.round_id, precrash.tag.round_id)
        self.assertGreater(resumed.outbound.tag.version, precrash.tag.version)

        stale = resumed.protocol.handle_verify_reply(VerifyReply(precrash.tag, hit=True))
        self.assertTrue(stale.ignored)
        completed = resumed.protocol.handle_recovery_reply(
            RecoveryReply(resumed.outbound.tag, finished=False)
        )
        self.assertTrue(completed.applied)

        stable = FissionProtocol.resume_after_crash(resumed.protocol.snapshot())
        self.assertIsNone(stable.outbound)
        self.assertEqual(stable.protocol.state, ProtocolState.READY_BACKUP)

    def test_local_transition_guards(self) -> None:
        protocol = FissionProtocol("r")
        with self.assertRaises(InvalidTransitionError):
            protocol.retry_verification()
        with self.assertRaises(InvalidTransitionError):
            protocol.invalidate_inflight_for_recovery()
        protocol.start_verification(round_id=2)
        with self.assertRaises(InvalidTransitionError):
            protocol.start_verification(round_id=3)
        with self.assertRaises(InvalidTransitionError):
            protocol.finish()
        protocol.handle_verify_reply(VerifyReply(MessageTag("r", 2, 1), hit=True))
        with self.assertRaises(InvalidTransitionError):
            protocol.start_verification(round_id=2)
        protocol.finish()
        self.assertEqual(protocol.state, ProtocolState.FINISHED)
        with self.assertRaises(InvalidTransitionError):
            protocol.start_verification()

    def test_request_can_finish_before_its_first_speculative_round(self) -> None:
        protocol = FissionProtocol("empty")
        protocol.finish()
        self.assertEqual(protocol.state, ProtocolState.FINISHED)
        restored = FissionProtocol.from_snapshot(protocol.snapshot())
        self.assertEqual(restored.state, ProtocolState.FINISHED)
        self.assertEqual(restored.round_id, -1)

    def test_message_validation_and_wrong_handler_types(self) -> None:
        invalid_tags = [
            (True, 0, 1),
            ("r", -1, 1),
            ("r", 0, 0),
            ("r", 0, True),
        ]
        for args in invalid_tags:
            with self.subTest(args=args), self.assertRaises(InvalidMessageError):
                MessageTag(*args)
        tag = MessageTag("r", 0, 1)
        with self.assertRaises(InvalidMessageError):
            VerifyReply(tag, hit=1)  # type: ignore[arg-type]
        with self.assertRaises(InvalidMessageError):
            RecoveryReply(tag, recovered_blocks=-1)

        protocol = FissionProtocol("r")
        with self.assertRaises(InvalidMessageError):
            protocol.handle_verify_reply(RecoveryReply(tag))  # type: ignore[arg-type]
        with self.assertRaises(InvalidMessageError):
            protocol.handle_recovery_reply(VerifyReply(tag, True))  # type: ignore[arg-type]

    def test_corrupt_snapshots_are_rejected(self) -> None:
        protocol = FissionProtocol("r")
        request = protocol.start_verification()
        snapshot = protocol.snapshot()
        corruptions = [
            replace(snapshot, schema_version=99),
            replace(snapshot, schema_version=True),
            replace(snapshot, active_tag=None),
            replace(snapshot, active_tag=MessageTag("r", 0, 2)),
            replace(snapshot, version=0),
            replace(snapshot, round_id=-1),
            replace(snapshot, state=ProtocolState.READY_HIT),
            replace(snapshot, accepted_blocks=-1),
        ]
        for corrupted in corruptions:
            with self.subTest(corrupted=corrupted), self.assertRaises(ProtocolSnapshotError):
                FissionProtocol.from_snapshot(corrupted)
        self.assertEqual(request.tag, protocol.active_tag)

    def test_concurrent_duplicate_delivery_applies_exactly_once(self) -> None:
        protocol = FissionProtocol("r")
        current = protocol.start_verification(round_id=8)
        replies = [VerifyReply(current.tag, hit=True) for _ in range(40)]
        replies.extend(
            VerifyReply(MessageTag("r", 8, version), hit=True) for version in range(2, 22)
        )
        random.Random(44).shuffle(replies)
        with ThreadPoolExecutor(max_workers=12) as executor:
            transitions = list(executor.map(protocol.handle_verify_reply, replies))
        self.assertEqual(sum(result.applied for result in transitions), 1)
        self.assertEqual(protocol.state, ProtocolState.READY_HIT)

    def test_randomized_delayed_reply_sequences_preserve_current_epoch(self) -> None:
        rng = random.Random(20260722)
        protocol = FissionProtocol("request")
        historical_verify: list[VerifyReply] = []
        historical_recovery: list[RecoveryReply] = []

        for round_id in range(300):
            request = protocol.start_verification(round_id)
            for _ in range(rng.randrange(3)):
                historical_verify.append(VerifyReply(request.tag, hit=True))
                request = protocol.retry_verification()

            # Old replies sampled from arbitrary prior epochs must be harmless.
            rng.shuffle(historical_verify)
            for stale in historical_verify[-5:]:
                result = protocol.handle_verify_reply(stale)
                self.assertTrue(result.ignored)
                self.assertEqual(protocol.active_tag, request.tag)

            hit = rng.random() < 0.55
            accepted = rng.randrange(9)
            exact = protocol.handle_verify_reply(
                VerifyReply(request.tag, hit=hit, accepted_blocks=accepted)
            )
            self.assertTrue(exact.applied)
            historical_verify.append(VerifyReply(request.tag, hit=hit))

            if not hit:
                recovery_tag = request.tag
                if rng.random() < 0.25:
                    old_tag = recovery_tag
                    recovery_tag = protocol.invalidate_inflight_for_recovery().tag
                    historical_recovery.append(RecoveryReply(old_tag))
                rng.shuffle(historical_recovery)
                for stale in historical_recovery[-5:]:
                    result = protocol.handle_recovery_reply(stale)
                    self.assertTrue(result.ignored)
                    self.assertEqual(protocol.active_tag, recovery_tag)
                recovered = protocol.handle_recovery_reply(RecoveryReply(recovery_tag))
                self.assertTrue(recovered.applied)
                historical_recovery.append(RecoveryReply(recovery_tag))

            self.assertIn(protocol.state, {ProtocolState.READY_HIT, ProtocolState.READY_BACKUP})
            # Round-trip validation is an inexpensive invariant audit.
            protocol = FissionProtocol.from_snapshot(protocol.snapshot())


if __name__ == "__main__":
    unittest.main()
