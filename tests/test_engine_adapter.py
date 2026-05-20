"""CPU-only engine seam, physical accounting, and state-space tests."""

from __future__ import annotations

import itertools
import unittest

from fissionspec.coordinator import (
    InMemoryCoordinator,
    InvalidCoordinatorOperation,
    PhysicalPageRef,
    SchedulerLane,
)
from fissionspec.engine_adapter import (
    FakeBlockDescriptor,
    GraphBucket,
    GraphBucketSet,
    GraphBucketUnavailableError,
    MockEngineAdapter,
    MockEngineHarness,
    NullEngineBackend,
    PhysicalBatch,
    ReservedEngineRow,
    VerifierOutcome,
)
from fissionspec.ledger import LedgerEpoch, OutOfPagesError
from fissionspec.protocol import MessageTag


def _new_harness(
    *,
    page_count: int = 32,
    page_size: int = 2,
    buckets: tuple[GraphBucket, ...] = (
        GraphBucket(1, 4, "one"),
        GraphBucket(2, 8, "two"),
        GraphBucket(4, 16, "four"),
    ),
) -> tuple[MockEngineHarness, NullEngineBackend]:
    backend = NullEngineBackend()
    adapter = MockEngineAdapter(
        InMemoryCoordinator(page_count, page_size),
        GraphBucketSet(buckets),
        backend,
    )
    return MockEngineHarness(adapter), backend


def _fake_row(request_id: int, width: int) -> ReservedEngineRow:
    tag = MessageTag(request_id, 0, 1)
    return ReservedEngineRow(
        FakeBlockDescriptor(
            request_id=request_id,
            tag=tag,
            epoch=LedgerEpoch(request_id, 0, 1),
            base_committed_blocks=0,
            reserved_blocks=width,
            pages=(PhysicalPageRef(request_id, 1),),
        ),
        verifier_slots=width,
    )


class PhysicalAccountingTests(unittest.TestCase):
    def test_graph_padding_is_never_hidden(self) -> None:
        harness, backend = _new_harness(buckets=(GraphBucket(4, 8, "padded-four-by-eight"),))
        for request_id in ("a", "b"):
            harness.adapter.register_request(request_id, total_blocks=2)
        rows = tuple(harness.adapter.reserve(request_id, 2) for request_id in ("a", "b"))
        batch = harness.adapter.submit_verify(rows)

        self.assertEqual(batch.logical_rows, 2)
        self.assertEqual(batch.physical_rows, 4)
        self.assertEqual(batch.padding_rows, 2)
        self.assertEqual(batch.useful_verifier_slots, 4)
        self.assertEqual(batch.physical_verifier_slots, 8)
        self.assertEqual(batch.padding_verifier_slots, 4)
        self.assertGreaterEqual(batch.padding_verifier_slots, batch.padding_rows)
        batch.audit_accounting()
        self.assertEqual(backend.verify_batches, (batch,))

    def test_bucket_rejects_shape_that_would_hide_a_padded_row(self) -> None:
        rows = (_fake_row(1, 3), _fake_row(2, 3), _fake_row(3, 2))
        bucket = GraphBucket(4, 8, "insufficient-after-row-padding")
        with self.assertRaises(ValueError):
            PhysicalBatch(rows, bucket)
        with self.assertRaises(GraphBucketUnavailableError):
            GraphBucketSet((bucket,)).choose(
                logical_rows=3,
                useful_verifier_slots=8,
            )

    def test_small_shape_domain_has_additive_accounting(self) -> None:
        for logical_rows in range(1, 4):
            for widths in itertools.product(range(1, 4), repeat=logical_rows):
                rows = tuple(_fake_row(index + 1, width) for index, width in enumerate(widths))
                useful_slots = sum(widths)
                for physical_rows in range(logical_rows, 5):
                    for physical_slots in range(physical_rows, 17):
                        with self.subTest(
                            logical_rows=logical_rows,
                            widths=widths,
                            physical_rows=physical_rows,
                            physical_slots=physical_slots,
                        ):
                            bucket = GraphBucket(
                                physical_rows,
                                physical_slots,
                                f"r{physical_rows}s{physical_slots}",
                            )
                            minimum = useful_slots + physical_rows - logical_rows
                            if physical_slots < minimum:
                                with self.assertRaises(ValueError):
                                    PhysicalBatch(rows, bucket)
                                continue
                            batch = PhysicalBatch(rows, bucket)
                            batch.audit_accounting()
                            self.assertEqual(
                                batch.physical_rows,
                                batch.logical_rows + batch.padding_rows,
                            )
                            self.assertEqual(
                                batch.physical_verifier_slots,
                                batch.useful_verifier_slots + batch.padding_verifier_slots,
                            )


class EngineSeamFaultTests(unittest.TestCase):
    def test_exact_callback_before_physical_submission_is_rejected(self) -> None:
        harness, _ = _new_harness()
        harness.adapter.register_request("r", total_blocks=1)
        row = harness.adapter.reserve("r", 1)
        harness.enqueue_verify(row, VerifierOutcome(True, 1))
        with self.assertRaises(InvalidCoordinatorOperation):
            harness.deliver()
        self.assertEqual(harness.adapter.coordinator.lane("r"), SchedulerLane.VERIFYING)
        self.assertEqual(harness.adapter.coordinator.request("r").committed_blocks, 0)
        harness.adapter.audit()

    def test_dropped_callback_has_no_effect_and_exact_retry_applies(self) -> None:
        harness, _ = _new_harness()
        harness.adapter.register_request("r", total_blocks=1)
        row = harness.adapter.reserve("r", 1)
        harness.adapter.submit_verify((row,))
        callback = harness.enqueue_verify(row, VerifierOutcome(True, 1))
        self.assertEqual(harness.drop(), callback)
        self.assertEqual(harness.adapter.coordinator.lane("r"), SchedulerLane.VERIFYING)
        self.assertEqual(harness.adapter.coordinator.request("r").committed_blocks, 0)

        harness.enqueue_verify(row, VerifierOutcome(True, 1))
        self.assertTrue(harness.deliver().applied)
        self.assertEqual(harness.adapter.coordinator.lane("r"), SchedulerLane.FINISHED)
        self.assertEqual(harness.adapter.coordinator.request("r").committed_blocks, 1)
        harness.adapter.audit()

    def test_duplicate_stale_and_reordered_verify_callbacks_are_exactly_once(self) -> None:
        harness, _ = _new_harness()
        harness.adapter.register_request("r", total_blocks=2)
        row = harness.adapter.reserve("r", 1)
        harness.adapter.submit_verify((row,))

        harness.enqueue_verify(
            row,
            VerifierOutcome(True, 1),
            tag=harness.stale_tag(row.tag),
        )
        harness.enqueue_verify(row, VerifierOutcome(True, 1))
        harness.duplicate(1)
        harness.reorder((2, 0, 1))
        results = harness.deliver_all()

        self.assertEqual(sum(result.applied for result in results), 1)
        self.assertEqual(harness.adapter.coordinator.request("r").committed_blocks, 1)
        self.assertEqual(harness.adapter.coordinator.allocator.allocated_count, 1)
        harness.adapter.audit()

    def test_miss_recovery_duplicates_are_fenced_and_finish_once(self) -> None:
        harness, backend = _new_harness()
        harness.adapter.register_request("r", total_blocks=1)
        row = harness.adapter.reserve("r", 1)
        harness.adapter.submit_verify((row,))
        harness.enqueue_verify(row, VerifierOutcome(False, 1))
        harness.duplicate()
        first, duplicate = harness.deliver_all()
        self.assertTrue(first.applied)
        self.assertTrue(duplicate.ignored)
        self.assertEqual(len(backend.recoveries), 1)

        recovery = harness.adapter.coordinator.latest_recovery("r")
        self.assertIsNotNone(recovery)
        assert recovery is not None
        harness.enqueue_recovery(recovery, finished=True)
        harness.duplicate()
        harness.reorder((1, 0))
        first_recovery, duplicate_recovery = harness.deliver_all()
        self.assertTrue(first_recovery.applied)
        self.assertTrue(duplicate_recovery.ignored)
        view = harness.adapter.coordinator.request("r")
        self.assertEqual(view.lane, SchedulerLane.FINISHED)
        self.assertEqual(view.committed_blocks, 1)
        self.assertEqual(harness.adapter.coordinator.allocator.allocated_count, 1)
        harness.adapter.audit()

    def test_oom_is_inert_and_cancel_frees_then_fences_callback(self) -> None:
        harness, _ = _new_harness(
            page_count=1,
            page_size=1,
            buckets=(GraphBucket(1, 4, "one"),),
        )
        harness.adapter.register_request("r", total_blocks=2)
        before = harness.adapter.coordinator.snapshot_bytes()
        with self.assertRaises(OutOfPagesError):
            harness.adapter.reserve("r", 2)
        self.assertEqual(harness.adapter.coordinator.snapshot_bytes(), before)
        self.assertEqual(harness.adapter.active_rows, ())

        row = harness.adapter.reserve("r", 1)
        harness.adapter.submit_verify((row,))
        harness.enqueue_verify(row, VerifierOutcome(True, 1))
        self.assertTrue(harness.adapter.cancel("r"))
        self.assertEqual(harness.adapter.coordinator.allocator.allocated_count, 0)
        self.assertTrue(harness.deliver().ignored)
        self.assertEqual(harness.adapter.coordinator.lane("r"), SchedulerLane.CANCELLED)
        harness.adapter.audit()

    def test_two_request_reordering_preserves_ownership(self) -> None:
        harness, _ = _new_harness()
        for request_id in ("a", "b"):
            harness.adapter.register_request(request_id, total_blocks=1)
        rows = tuple(harness.adapter.reserve(request_id, 1) for request_id in ("a", "b"))
        harness.adapter.submit_verify(rows)
        harness.enqueue_verify(rows[0], VerifierOutcome(True, 1))
        harness.enqueue_verify(rows[1], VerifierOutcome(False, 1))
        harness.reorder((1, 0))
        results = harness.deliver_all()
        self.assertEqual(sum(result.applied for result in results), 2)

        recovery = harness.adapter.coordinator.latest_recovery("b")
        self.assertIsNotNone(recovery)
        assert recovery is not None
        harness.enqueue_recovery(recovery, finished=True)
        self.assertTrue(harness.deliver().applied)
        for request_id in ("a", "b"):
            self.assertEqual(
                harness.adapter.coordinator.lane(request_id),
                SchedulerLane.FINISHED,
            )
        self.assertEqual(harness.adapter.coordinator.allocator.allocated_count, 2)
        harness.adapter.audit()


class ExhaustiveSmallStateTests(unittest.TestCase):
    def test_all_small_outcome_prefixes_reach_terminal_state_with_exact_ownership(self) -> None:
        atoms = (
            VerifierOutcome(True, 1),
            VerifierOutcome(False, 1),
            VerifierOutcome(False, 0),
        )
        for page_size in (1, 2):
            for total_blocks in range(1, 4):
                for outcomes in itertools.product(atoms, repeat=3):
                    with self.subTest(
                        page_size=page_size,
                        total_blocks=total_blocks,
                        outcomes=outcomes,
                    ):
                        harness, backend = _new_harness(
                            page_count=16,
                            page_size=page_size,
                            buckets=(GraphBucket(1, 4, "one"),),
                        )
                        harness.adapter.register_request("r", total_blocks=total_blocks)
                        lane = harness.drive_to_terminal(
                            "r",
                            outcomes,
                            verification_width=1,
                            maximum_steps=32,
                        )
                        self.assertEqual(lane, SchedulerLane.FINISHED)
                        view = harness.adapter.coordinator.request("r")
                        self.assertEqual(view.committed_blocks, total_blocks)
                        expected_pages = (total_blocks + page_size - 1) // page_size
                        self.assertEqual(
                            harness.adapter.coordinator.allocator.allocated_count,
                            expected_pages,
                        )
                        self.assertEqual(harness.pending, ())
                        for batch in backend.verify_batches:
                            batch.audit_accounting()
                        harness.adapter.audit()


if __name__ == "__main__":
    unittest.main()
