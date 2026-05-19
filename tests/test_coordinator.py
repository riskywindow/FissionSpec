"""Composed protocol/ledger tests with exhaustive small-trace fault injection."""

from __future__ import annotations

import unittest

from fissionspec.coordinator import (
    CoordinatorCrash,
    CoordinatorSnapshotError,
    CrashAt,
    FakeRemoteDrafter,
    FakeTargetVerifier,
    FaultPoint,
    InMemoryCoordinator,
    InvalidCoordinatorOperation,
    SchedulerLane,
)
from fissionspec.ledger import OutOfPagesError
from fissionspec.protocol import (
    MessageTag,
    RecoveryReply,
    ReplyDisposition,
    VerifyReply,
)


def _drive_to_completion(coordinator: InMemoryCoordinator, request_id: str | int) -> list[int]:
    """Make one block of target-authorized progress per bounded step."""

    versions = [coordinator.request(request_id).protocol_version]
    for _ in range(32):
        view = coordinator.request(request_id)
        if view.lane is SchedulerLane.FINISHED:
            coordinator.audit()
            return versions
        if view.lane is SchedulerLane.RECOVERING:
            recovery = coordinator.latest_recovery(request_id)
            if recovery is None:
                raise AssertionError("recovering request has no recovery command")
            result = coordinator.on_recovery_complete(
                RecoveryReply(
                    recovery.tag,
                    finished=view.committed_blocks == view.total_blocks,
                )
            )
            if not result.applied:
                raise AssertionError("latest recovery unexpectedly became stale")
        elif view.lane in {
            SchedulerLane.READY,
            SchedulerLane.READY_HIT,
            SchedulerLane.READY_BACKUP,
        }:
            dispatch = coordinator.reserve(request_id, 1)
            result = coordinator.on_verify_complete(
                FakeTargetVerifier.reply(
                    dispatch,
                    hit=True,
                    accepted_blocks=1,
                )
            )
            if not result.applied:
                raise AssertionError("exact verifier reply unexpectedly became stale")
        else:
            raise AssertionError(f"cannot drive request from {view.lane.value}")
        current = coordinator.request(request_id).protocol_version
        if current < versions[-1]:
            raise AssertionError("protocol version moved backwards")
        versions.append(current)
        coordinator.audit()
    raise AssertionError("request did not eventually complete")


class CoordinatorCompositionTests(unittest.TestCase):
    def test_exactly_once_hit_commits_and_finishes(self) -> None:
        coordinator = InMemoryCoordinator(page_count=8, page_size=2)
        coordinator.register_request("r", total_blocks=3)
        self.assertEqual(coordinator.next_batch(), ("r",))

        first = coordinator.reserve("r", 2)
        self.assertEqual(first.block_table.base_committed_blocks, 0)
        self.assertEqual(first.block_table.reserved_blocks, 2)
        self.assertEqual(coordinator.lane("r"), SchedulerLane.VERIFYING)
        reply = FakeTargetVerifier.reply(first, hit=True, accepted_blocks=2)
        applied = coordinator.on_verify_complete(reply)
        self.assertTrue(applied.applied)
        self.assertEqual(applied.lane, SchedulerLane.READY_HIT)
        self.assertEqual(coordinator.request("r").committed_blocks, 2)

        duplicate = coordinator.on_verify_complete(reply)
        self.assertTrue(duplicate.ignored)
        self.assertEqual(coordinator.request("r").committed_blocks, 2)
        self.assertEqual(coordinator.allocator.allocated_count, 1)

        second = coordinator.reserve("r", 1)
        finished = coordinator.on_verify_complete(
            FakeTargetVerifier.reply(second, hit=True, accepted_blocks=1)
        )
        self.assertEqual(finished.lane, SchedulerLane.FINISHED)
        self.assertEqual(coordinator.request("r").committed_blocks, 3)
        self.assertEqual(coordinator.allocator.allocated_count, 2)
        self.assertEqual(coordinator.next_batch(), ())
        coordinator.audit()

    def test_drop_duplicate_and_reorder_across_two_recoveries(self) -> None:
        coordinator = InMemoryCoordinator(page_count=16, page_size=2)
        coordinator.register_request("a", total_blocks=2)
        coordinator.register_request("b", total_blocks=2)
        dispatches = {request_id: coordinator.reserve(request_id, 1) for request_id in ("a", "b")}

        # Reorder target replies. Both accepted prefixes commit independently.
        recoveries = {}
        for request_id in ("b", "a"):
            result = coordinator.on_verify_complete(
                FakeTargetVerifier.reply(
                    dispatches[request_id],
                    hit=False,
                    accepted_blocks=1,
                )
            )
            self.assertTrue(result.applied)
            self.assertIsNotNone(result.outbound)
            assert result.outbound is not None
            recoveries[request_id] = result.outbound
            self.assertTrue(
                coordinator.on_verify_complete(
                    FakeTargetVerifier.reply(
                        dispatches[request_id],
                        hit=False,
                        accepted_blocks=1,
                    )
                ).ignored
            )

        drafter = FakeRemoteDrafter()
        drafter.submit(recoveries["a"])
        drafter.submit(recoveries["b"])
        drafter.duplicate(0)
        drafter.reorder([1, 2, 0])  # b, a, duplicated a
        dropped = drafter.drop(0)
        self.assertEqual(dropped.tag.request_id, "b")

        first_a = coordinator.on_recovery_complete(drafter.reply(0, finished=False))
        duplicate_a = coordinator.on_recovery_complete(drafter.reply(0, finished=False))
        self.assertTrue(first_a.applied)
        self.assertTrue(duplicate_a.ignored)

        # Retry the dropped transport command. Its identity is still current.
        latest_b = coordinator.latest_recovery("b")
        self.assertIsNotNone(latest_b)
        assert latest_b is not None
        drafter.submit(latest_b)
        self.assertTrue(coordinator.on_recovery_complete(drafter.reply(0, finished=False)).applied)
        self.assertEqual(coordinator.next_batch(), ("a", "b"))

        for request_id in ("a", "b"):
            versions = _drive_to_completion(coordinator, request_id)
            self.assertEqual(versions, sorted(versions))
        self.assertEqual(coordinator.allocator.allocated_count, 2)
        coordinator.audit()

    def test_stale_unknown_and_semantically_invalid_callbacks_are_inert(self) -> None:
        coordinator = InMemoryCoordinator(page_count=8, page_size=2)
        coordinator.register_request(7, total_blocks=2)
        dispatch = coordinator.reserve(7, 1)
        baseline = coordinator.snapshot_bytes()

        stale = VerifyReply(
            MessageTag(7, dispatch.request.tag.round_id, dispatch.request.tag.version + 1),
            hit=True,
            accepted_blocks=1,
        )
        self.assertEqual(
            coordinator.on_verify_complete(stale).disposition,
            ReplyDisposition.IGNORED_STALE,
        )
        unknown = VerifyReply(MessageTag("unknown", 0, 1), hit=True)
        self.assertTrue(coordinator.on_verify_complete(unknown).ignored)
        self.assertEqual(coordinator.snapshot_bytes(), baseline)

        with self.assertRaises(InvalidCoordinatorOperation):
            coordinator.on_verify_complete(
                FakeTargetVerifier.reply(
                    dispatch,
                    hit=True,
                    accepted_blocks=2,
                )
            )
        self.assertEqual(coordinator.snapshot_bytes(), baseline)

        miss = coordinator.on_verify_complete(
            FakeTargetVerifier.reply(
                dispatch,
                hit=False,
                accepted_blocks=1,
            )
        )
        assert miss.outbound is not None
        recovery_baseline = coordinator.snapshot_bytes()
        with self.assertRaises(InvalidCoordinatorOperation):
            coordinator.on_recovery_complete(RecoveryReply(miss.outbound.tag, finished=True))
        self.assertEqual(coordinator.snapshot_bytes(), recovery_baseline)
        self.assertTrue(
            coordinator.on_recovery_complete(
                RecoveryReply(miss.outbound.tag, finished=False)
            ).applied
        )
        coordinator.audit()

    def test_oom_preflight_is_stable_and_cancel_releases_every_page(self) -> None:
        coordinator = InMemoryCoordinator(page_count=1, page_size=1)
        coordinator.register_request("oom", total_blocks=2)
        baseline = coordinator.snapshot_bytes()
        with self.assertRaises(OutOfPagesError):
            coordinator.reserve("oom", 2)
        self.assertEqual(coordinator.snapshot_bytes(), baseline)
        self.assertEqual(coordinator.request("oom").round_id, -1)
        self.assertEqual(coordinator.request("oom").protocol_version, 0)
        self.assertEqual(coordinator.allocator.allocated_count, 0)

        dispatch = coordinator.reserve("oom", 1)
        self.assertEqual(coordinator.allocator.allocated_count, 1)
        self.assertTrue(coordinator.cancel("oom"))
        self.assertFalse(coordinator.cancel("oom"))
        self.assertEqual(coordinator.lane("oom"), SchedulerLane.CANCELLED)
        self.assertEqual(coordinator.allocator.allocated_count, 0)
        self.assertEqual(coordinator.allocator.free_count, 1)
        self.assertTrue(
            coordinator.on_verify_complete(
                FakeTargetVerifier.reply(
                    dispatch,
                    hit=True,
                    accepted_blocks=1,
                )
            ).ignored
        )
        coordinator.audit()

    def test_crash_resume_fences_precrash_replies(self) -> None:
        coordinator = InMemoryCoordinator(page_count=8, page_size=2)
        coordinator.register_request("r", total_blocks=2)
        dispatch = coordinator.reserve("r", 1)
        durable = coordinator.snapshot_bytes()
        old_version = dispatch.request.tag.version

        restored = InMemoryCoordinator.from_snapshot_bytes(durable)
        resumed = restored.request("r")
        self.assertEqual(resumed.lane, SchedulerLane.RECOVERING)
        self.assertGreater(resumed.protocol_version, old_version)
        self.assertEqual(restored.allocator.allocated_count, 0)
        self.assertTrue(
            restored.on_verify_complete(
                FakeTargetVerifier.reply(
                    dispatch,
                    hit=True,
                    accepted_blocks=1,
                )
            ).ignored
        )

        old_recovery = RecoveryReply(dispatch.request.tag, finished=False)
        self.assertTrue(restored.on_recovery_complete(old_recovery).ignored)
        latest = restored.latest_recovery("r")
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertTrue(
            restored.on_recovery_complete(RecoveryReply(latest.tag, finished=False)).applied
        )
        versions = _drive_to_completion(restored, "r")
        self.assertEqual(versions, sorted(versions))
        self.assertEqual(restored.allocator.allocated_count, 1)

    def test_snapshot_is_canonical_checked_and_preserves_identifier_types(self) -> None:
        coordinator = InMemoryCoordinator(page_count=8, page_size=2)
        coordinator.register_request(1, total_blocks=1)
        coordinator.register_request("1", total_blocks=1)
        coordinator.reserve(1, 1)
        snapshot = coordinator.snapshot_bytes()
        self.assertEqual(snapshot, coordinator.snapshot_bytes())

        exact = InMemoryCoordinator.from_snapshot_bytes(snapshot, resume_after_crash=False)
        self.assertEqual({view.request_id for view in exact.requests()}, {1, "1"})
        self.assertEqual(exact.lane(1), SchedulerLane.VERIFYING)
        exact.audit()

        tampered_checksum = bytearray(snapshot)
        checksum_start = snapshot.index(b'"checksum":"') + len(b'"checksum":"')
        tampered_checksum[checksum_start] = (
            ord("0") if tampered_checksum[checksum_start] != ord("0") else ord("1")
        )
        malformed = [
            bytes(tampered_checksum),
            snapshot[:-1],
            snapshot[:-2] + b"]\n",
            snapshot[:-1] + b" \n",
            (b'{"checksum":"0","checksum":"0","payload":{},"schema_version":1}\n'),
        ]
        for payload in malformed:
            with (
                self.subTest(payload=payload[:32]),
                self.assertRaises(CoordinatorSnapshotError),
            ):
                InMemoryCoordinator.from_snapshot_bytes(payload)


class CoordinatorFaultInjectionTests(unittest.TestCase):
    def _new_ready(self) -> InMemoryCoordinator:
        coordinator = InMemoryCoordinator(page_count=8, page_size=2)
        coordinator.register_request("r", total_blocks=2)
        return coordinator

    def _assert_terminal_ownership(self, coordinator: InMemoryCoordinator) -> None:
        view = coordinator.request("r")
        self.assertEqual(view.lane, SchedulerLane.FINISHED)
        self.assertEqual(view.committed_blocks, view.total_blocks)
        self.assertEqual(coordinator.allocator.allocated_count, 1)
        self.assertEqual(coordinator.allocator.free_count, 7)
        coordinator.audit()

    def test_every_reserve_fault_replays_to_eventual_completion(self) -> None:
        points = {
            FaultPoint.BEFORE_RESERVE,
            FaultPoint.AFTER_LEDGER_STAGE,
            FaultPoint.AFTER_PROTOCOL_START,
        }
        for point in points:
            with self.subTest(point=point):
                coordinator = self._new_ready()
                durable = coordinator.snapshot_bytes()
                injector = CrashAt(point)
                coordinator.set_fault_hook(injector)
                with self.assertRaises(CoordinatorCrash):
                    coordinator.reserve("r", 1)
                self.assertTrue(injector.fired)

                restored = InMemoryCoordinator.from_snapshot_bytes(durable)
                versions = _drive_to_completion(restored, "r")
                self.assertEqual(versions, sorted(versions))
                self._assert_terminal_ownership(restored)

    def test_every_verify_fault_fences_old_reply_and_completes_once(self) -> None:
        points = {
            FaultPoint.BEFORE_VERIFY_COMMIT,
            FaultPoint.AFTER_LEDGER_COMMIT,
            FaultPoint.AFTER_PROTOCOL_TRANSITION,
        }
        for point in points:
            with self.subTest(point=point):
                coordinator = self._new_ready()
                dispatch = coordinator.reserve("r", 1)
                reply = FakeTargetVerifier.reply(
                    dispatch,
                    hit=True,
                    accepted_blocks=1,
                )
                durable = coordinator.snapshot_bytes()
                injector = CrashAt(point)
                coordinator.set_fault_hook(injector)
                with self.assertRaises(CoordinatorCrash):
                    coordinator.on_verify_complete(reply)
                self.assertTrue(injector.fired)

                restored = InMemoryCoordinator.from_snapshot_bytes(durable)
                resumed = restored.request("r")
                self.assertGreater(resumed.protocol_version, dispatch.request.tag.version)
                before = resumed.committed_blocks
                self.assertTrue(restored.on_verify_complete(reply).ignored)
                self.assertEqual(restored.request("r").committed_blocks, before)
                versions = _drive_to_completion(restored, "r")
                self.assertEqual(versions, sorted(versions))
                self._assert_terminal_ownership(restored)

    def test_every_recovery_fault_fences_old_reply_and_completes_once(self) -> None:
        points = {
            FaultPoint.BEFORE_RECOVERY_APPLY,
            FaultPoint.AFTER_RECOVERY_APPLY,
        }
        for point in points:
            with self.subTest(point=point):
                coordinator = self._new_ready()
                dispatch = coordinator.reserve("r", 1)
                miss = coordinator.on_verify_complete(
                    FakeTargetVerifier.reply(
                        dispatch,
                        hit=False,
                        accepted_blocks=1,
                    )
                )
                assert miss.outbound is not None
                old_reply = RecoveryReply(miss.outbound.tag, finished=False)
                durable = coordinator.snapshot_bytes()
                injector = CrashAt(point)
                coordinator.set_fault_hook(injector)
                with self.assertRaises(CoordinatorCrash):
                    coordinator.on_recovery_complete(old_reply)
                self.assertTrue(injector.fired)

                restored = InMemoryCoordinator.from_snapshot_bytes(durable)
                self.assertGreater(
                    restored.request("r").protocol_version,
                    miss.outbound.tag.version,
                )
                before = restored.request("r").committed_blocks
                self.assertTrue(restored.on_recovery_complete(old_reply).ignored)
                self.assertEqual(restored.request("r").committed_blocks, before)
                versions = _drive_to_completion(restored, "r")
                self.assertEqual(versions, sorted(versions))
                self._assert_terminal_ownership(restored)

    def test_every_cancel_fault_replays_to_terminal_page_free_state(self) -> None:
        for point in {FaultPoint.BEFORE_CANCEL, FaultPoint.AFTER_CANCEL}:
            with self.subTest(point=point):
                coordinator = self._new_ready()
                dispatch = coordinator.reserve("r", 1)
                durable = coordinator.snapshot_bytes()
                injector = CrashAt(point)
                coordinator.set_fault_hook(injector)
                with self.assertRaises(CoordinatorCrash):
                    coordinator.cancel("r")
                self.assertTrue(injector.fired)

                restored = InMemoryCoordinator.from_snapshot_bytes(durable)
                self.assertTrue(
                    restored.on_verify_complete(
                        FakeTargetVerifier.reply(
                            dispatch,
                            hit=True,
                            accepted_blocks=1,
                        )
                    ).ignored
                )
                prior_version = restored.request("r").protocol_version
                self.assertTrue(restored.cancel("r"))
                view = restored.request("r")
                self.assertGreaterEqual(view.protocol_version, prior_version)
                self.assertEqual(view.lane, SchedulerLane.CANCELLED)
                self.assertEqual(restored.allocator.allocated_count, 0)
                self.assertEqual(restored.allocator.free_count, 8)
                restored.audit()


if __name__ == "__main__":
    unittest.main()
