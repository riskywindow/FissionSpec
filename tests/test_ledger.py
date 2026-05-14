"""Invariant, ABA, COW-tail, and crash-recovery tests for the KV ledger."""

from __future__ import annotations

import random
import unittest
from dataclasses import replace
from multiprocessing import get_all_start_methods, get_context
from multiprocessing.connection import Connection

from fissionspec.ledger import (
    AllocatorSnapshot,
    DoubleReleaseError,
    DuplicateOutcomeError,
    DuplicateRequestError,
    FixedPageAllocator,
    ForeignPageHandleError,
    ForkedAllocatorError,
    InvalidCommitError,
    InvalidConfigurationError,
    InvalidPageHandleError,
    InvariantViolation,
    LedgerEpoch,
    OutcomeNotFoundError,
    OutOfPagesError,
    PageHandle,
    PageSpan,
    PoolIdentity,
    SnapshotError,
    SpeculativeLedger,
    StaleEpochError,
    StalePageHandleError,
    TransactionConflictError,
)


def _fork_allocator_probe(
    inherited: FixedPageAllocator,
    snapshot: AllocatorSnapshot,
    sender: Connection,
) -> None:
    """Exercise inherited and restored pools in a forked child process."""

    try:
        inherited.allocate()
    except Exception as exc:
        inherited_error = type(exc).__name__
    else:
        inherited_error = "none"
    restored = FixedPageAllocator.from_snapshot(snapshot)
    restored_handle = next(iter(restored.live_handles()))
    sender.send(
        (
            inherited_error,
            restored.pool_id,
            restored_handle.pool_id,
            restored.owner_of(restored_handle),
        )
    )
    sender.close()


class FixedPageAllocatorTests(unittest.TestCase):
    def test_finite_capacity_and_deterministic_lowest_page_allocation(self) -> None:
        allocator = FixedPageAllocator(page_count=3, page_size=16)
        handles = [allocator.allocate(f"owner-{index}") for index in range(3)]
        self.assertEqual([handle.page_id for handle in handles], [0, 1, 2])
        self.assertEqual(allocator.free_count, 0)
        with self.assertRaises(OutOfPagesError):
            allocator.allocate()
        allocator.release(handles[1])
        reused = allocator.allocate("new-owner")
        self.assertEqual(reused.page_id, 1)
        self.assertEqual(reused.generation, handles[1].generation + 1)
        allocator.audit()

    def test_generation_fences_stale_aba_release(self) -> None:
        allocator = FixedPageAllocator(1, 4)
        old = allocator.allocate("old")
        allocator.release(old)
        with self.assertRaises(DoubleReleaseError):
            allocator.release(old)

        current = allocator.allocate("current")
        self.assertEqual(old.page_id, current.page_id)
        self.assertNotEqual(old.generation, current.generation)
        with self.assertRaises(StalePageHandleError):
            allocator.release(old)
        self.assertTrue(allocator.is_live(current))
        self.assertEqual(allocator.owner_of(current), "current")
        allocator.release(current)
        allocator.audit()

    def test_invalid_page_and_stale_clone_are_rejected(self) -> None:
        allocator = FixedPageAllocator(2, 4)
        source = allocator.allocate()
        allocator.release(source)
        with self.assertRaises(StalePageHandleError):
            allocator.clone(source)
        with self.assertRaises(InvalidPageHandleError):
            allocator.release(PageHandle(2, 1, allocator.pool_id))

    def test_handles_from_numerically_identical_foreign_pool_are_rejected(self) -> None:
        first = FixedPageAllocator(2, 4)
        second = FixedPageAllocator(2, 4)
        first_handle = first.allocate("first")
        second_handle = second.allocate("second")
        self.assertEqual(
            (first_handle.page_id, first_handle.generation),
            (second_handle.page_id, second_handle.generation),
        )
        self.assertNotEqual(first_handle.pool_id, second_handle.pool_id)

        foreign_operations = [
            lambda: second.release(first_handle),
            lambda: second.clone(first_handle),
            lambda: second.reassign(first_handle, "forged"),
            lambda: second.owner_of(first_handle),
            lambda: second.is_live(first_handle),
        ]
        for operation in foreign_operations:
            with self.subTest(operation=operation), self.assertRaises(ForeignPageHandleError):
                operation()
        self.assertTrue(first.is_live(first_handle))
        self.assertTrue(second.is_live(second_handle))
        first.audit()
        second.audit()

    def test_snapshot_round_trip_preserves_generation_and_free_partition(self) -> None:
        allocator = FixedPageAllocator(4, 8)
        first = allocator.allocate("first")
        second = allocator.allocate("second")
        allocator.release(first)
        allocator.reassign(second, "moved")
        snapshot = allocator.snapshot()

        restored = FixedPageAllocator.from_snapshot(snapshot)
        restored_snapshot = restored.snapshot()
        self.assertEqual(allocator.snapshot(), snapshot)
        self.assertNotEqual(restored.pool_id, snapshot.pool_id)
        self.assertEqual(restored_snapshot.generations, snapshot.generations)
        self.assertEqual(restored_snapshot.free_pages, snapshot.free_pages)
        self.assertEqual(
            tuple(
                (handle.page_id, handle.generation, owner)
                for handle, owner in restored_snapshot.live
            ),
            tuple((handle.page_id, handle.generation, owner) for handle, owner in snapshot.live),
        )
        restored_second = next(
            handle for handle in restored.live_handles() if handle.page_id == second.page_id
        )
        self.assertEqual(restored.owner_of(restored_second), "moved")
        with self.assertRaises(ForeignPageHandleError):
            restored.owner_of(second)
        reused = restored.allocate()
        self.assertEqual(reused.page_id, first.page_id)
        self.assertEqual(reused.generation, first.generation + 1)
        restored.audit()

        concurrent_restore = FixedPageAllocator.from_snapshot(snapshot)
        self.assertNotEqual(concurrent_restore.pool_id, restored.pool_id)
        with self.assertRaises(ForeignPageHandleError):
            concurrent_restore.is_live(restored_second)
        with self.assertRaises(ForeignPageHandleError):
            allocator.is_live(restored_second)

    def test_in_place_restore_invalidates_both_source_and_destination_handles(self) -> None:
        source = FixedPageAllocator(2, 4)
        source_handle = source.allocate("source")
        snapshot = source.snapshot()

        destination = FixedPageAllocator(3, 8)
        destination_handle = destination.allocate("destination")
        old_destination_pool = destination.pool_id
        destination.restore(snapshot)

        self.assertNotEqual(destination.pool_id, snapshot.pool_id)
        self.assertNotEqual(destination.pool_id, old_destination_pool)
        for foreign in (source_handle, destination_handle):
            with self.subTest(foreign=foreign), self.assertRaises(ForeignPageHandleError):
                destination.is_live(foreign)
        restored_handle = next(iter(destination.live_handles()))
        self.assertEqual(destination.owner_of(restored_handle), "source")
        self.assertEqual(destination.page_count, source.page_count)
        self.assertEqual(destination.page_size, source.page_size)
        destination.audit()

    def test_corrupt_allocator_snapshots_fail_closed(self) -> None:
        allocator = FixedPageAllocator(3, 4)
        live = allocator.allocate()
        snapshot = allocator.snapshot()
        corruptions = [
            replace(snapshot, schema_version=99),
            replace(snapshot, schema_version=True),
            replace(
                snapshot,
                free_pages=(*snapshot.free_pages, snapshot.free_pages[0]),
            ),
            replace(snapshot, free_pages=(True, 2)),
            replace(snapshot, generations=(0,)),
            replace(snapshot, live=(snapshot.live[0], snapshot.live[0])),
            replace(
                snapshot,
                live=(
                    (
                        PageHandle(
                            live.page_id,
                            live.generation + 1,
                            allocator.pool_id,
                        ),
                        None,
                    ),
                ),
            ),
            replace(snapshot, pool_id=FixedPageAllocator(1, 1).pool_id),
            replace(
                snapshot,
                live=(
                    (
                        PageHandle(
                            live.page_id,
                            live.generation,
                            FixedPageAllocator(1, 1).pool_id,
                        ),
                        None,
                    ),
                ),
            ),
            AllocatorSnapshot(
                page_count=3,
                page_size=4,
                pool_id=snapshot.pool_id,
                generations=(1, 0, 0),
                live=(),
                free_pages=(1, 2),
            ),
        ]
        for corrupted in corruptions:
            with self.subTest(corrupted=corrupted), self.assertRaises(SnapshotError):
                FixedPageAllocator.from_snapshot(corrupted)

    def test_configuration_and_handle_validation(self) -> None:
        for page_count, page_size in ((0, 4), (2, 0), (-1, 3), (True, 3)):
            with (
                self.subTest(page_count=page_count, page_size=page_size),
                self.assertRaises(InvalidConfigurationError),
            ):
                FixedPageAllocator(page_count, page_size)
        pool_id = FixedPageAllocator(1, 1).pool_id
        for args in ((-1, 1, pool_id), (0, 0, pool_id), (True, 1, pool_id)):
            with self.subTest(args=args), self.assertRaises(InvalidPageHandleError):
                PageHandle(*args)
        with self.assertRaises(InvalidPageHandleError):
            PageHandle(0, 1, object())  # type: ignore[arg-type]
        with self.assertRaises(InvalidConfigurationError):
            PoolIdentity("not-a-session", 1, 1)

    def test_pool_identities_are_unique_within_the_process_session(self) -> None:
        pools = {FixedPageAllocator(1, 1).pool_id for _ in range(512)}
        self.assertEqual(len(pools), 512)

    def test_forked_allocator_must_be_restored_into_fresh_child_pool(self) -> None:
        if "fork" not in get_all_start_methods():
            self.skipTest("platform has no fork start method")
        allocator = FixedPageAllocator(2, 4)
        allocator.allocate("persisted")
        snapshot = allocator.snapshot()
        context = get_context("fork")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_fork_allocator_probe,
            args=(allocator, snapshot, sender),
        )
        process.start()
        sender.close()
        self.assertTrue(receiver.poll(5.0), "forked allocator probe timed out")
        inherited_error, child_pool, handle_pool, owner = receiver.recv()
        receiver.close()
        process.join(timeout=5.0)
        self.assertEqual(process.exitcode, 0)
        self.assertEqual(inherited_error, ForkedAllocatorError.__name__)
        self.assertNotEqual(child_pool, allocator.pool_id)
        self.assertEqual(handle_pool, child_pool)
        self.assertEqual(owner, "persisted")


class SpeculativeLedgerTests(unittest.TestCase):
    def make_ledger(self, pages: int = 32, page_size: int = 4) -> SpeculativeLedger:
        return SpeculativeLedger(FixedPageAllocator(pages, page_size))

    def assert_aligned(self, ledger: SpeculativeLedger, request_id: str | int) -> None:
        view = ledger.request(request_id)
        self.assertEqual(sum(page.used for page in view.committed_pages), view.committed_blocks)
        for index, page in enumerate(view.committed_pages):
            self.assertEqual(page.start, index * ledger.page_size)
            if index + 1 < len(view.committed_pages):
                self.assertEqual(page.used, ledger.page_size)

    def test_partial_tail_is_cow_per_outcome_and_nonzero_commit_replaces_it(self) -> None:
        ledger = self.make_ledger()
        initial = ledger.register_request("req", committed_blocks=6)
        old_full, old_tail = initial.committed_pages
        epoch = ledger.begin("req", 0)
        alpha = ledger.stage_outcome(epoch, "alpha", appended_blocks=4)
        beta = ledger.stage_outcome(epoch, "beta", appended_blocks=3)

        self.assertTrue(alpha.cow_tail)
        self.assertTrue(beta.cow_tail)
        self.assertEqual([(page.start, page.used) for page in alpha.pages], [(4, 4), (8, 2)])
        self.assertEqual([(page.start, page.used) for page in beta.pages], [(4, 4), (8, 1)])
        self.assertNotEqual(alpha.pages[0].handle, beta.pages[0].handle)
        self.assertNotEqual(alpha.pages[0].handle, old_tail.handle)

        result = ledger.commit(epoch, "alpha", accepted_blocks=1)
        committed = result.request
        self.assertEqual(committed.committed_blocks, 7)
        self.assertEqual(committed.committed_pages[0], old_full)
        self.assertEqual(committed.committed_pages[1].handle, alpha.pages[0].handle)
        self.assertEqual(committed.committed_pages[1].used, 3)
        self.assertFalse(ledger.allocator.is_live(old_tail.handle))
        self.assertFalse(ledger.allocator.is_live(alpha.pages[1].handle))
        self.assertFalse(ledger.allocator.is_live(beta.pages[0].handle))
        self.assertEqual(ledger.allocator.allocated_count, 2)
        ledger.audit()

        # Reusing the old physical id proves that its delayed release cannot
        # free the new generation (the classic allocator ABA failure).
        current = ledger.allocator.allocate("temporary external check")
        self.assertEqual(current.page_id, old_tail.handle.page_id)
        with self.assertRaises(StalePageHandleError):
            ledger.allocator.release(old_tail.handle)
        ledger.allocator.release(current)
        ledger.audit()

    def test_zero_prefix_preserves_committed_partial_tail(self) -> None:
        ledger = self.make_ledger()
        before = ledger.register_request("r", 3)
        epoch = ledger.begin("r", 0)
        branch = ledger.stage_outcome(epoch, "miss", 7)
        result = ledger.commit(epoch, "miss", 0)
        self.assertEqual(result.request.committed_blocks, 3)
        self.assertEqual(result.request.committed_pages, before.committed_pages)
        self.assertTrue(ledger.allocator.is_live(before.committed_pages[-1].handle))
        self.assertTrue(all(not ledger.allocator.is_live(page.handle) for page in branch.pages))
        ledger.audit()

    def test_full_tail_prefix_commit_truncates_and_releases_suffix(self) -> None:
        ledger = self.make_ledger()
        initial = ledger.register_request("r", 8)
        epoch = ledger.begin("r", 10)
        branch = ledger.stage_outcome(epoch, "candidate", 9)
        self.assertFalse(branch.cow_tail)
        self.assertEqual(
            [(page.start, page.used) for page in branch.pages],
            [(8, 4), (12, 4), (16, 1)],
        )

        result = ledger.commit(epoch, "candidate", 5)
        self.assertEqual(result.committed_blocks, 13)
        self.assertEqual(result.request.committed_pages[:2], initial.committed_pages)
        self.assertEqual(
            [(page.start, page.used) for page in result.request.committed_pages],
            [(0, 4), (4, 4), (8, 4), (12, 1)],
        )
        self.assertFalse(ledger.allocator.is_live(branch.pages[-1].handle))
        self.assert_aligned(ledger, "r")
        ledger.audit()

    def test_abort_is_idempotent_but_other_stale_operations_are_rejected(self) -> None:
        ledger = self.make_ledger()
        ledger.register_request("r", 5)
        epoch = ledger.begin("r", 0)
        branch = ledger.stage_outcome(epoch, "x", 8)
        self.assertTrue(ledger.abort(epoch))
        self.assertFalse(ledger.abort(epoch))
        self.assertTrue(all(not ledger.allocator.is_live(page.handle) for page in branch.pages))
        with self.assertRaises(StaleEpochError):
            ledger.stage_outcome(epoch, "late", 1)
        with self.assertRaises(StaleEpochError):
            ledger.commit(epoch, "x", 0)
        with self.assertRaises(StaleEpochError):
            ledger.begin("r", 0)

        next_epoch = ledger.begin("r", 1)
        with self.assertRaises(TransactionConflictError):
            ledger.begin("r", 2)
        with self.assertRaises(StaleEpochError):
            ledger.stage_outcome(
                LedgerEpoch("r", next_epoch.round_id, next_epoch.version + 1),
                "future",
                1,
            )
        ledger.abort(next_epoch)
        ledger.audit()

    def test_retry_fences_same_round_and_releases_old_private_pages(self) -> None:
        ledger = self.make_ledger()
        ledger.register_request("r", 3)
        old_epoch = ledger.begin("r", 7)
        old_branch = ledger.stage_outcome(old_epoch, "old", 6)

        replacement = ledger.retry(old_epoch)
        self.assertEqual(replacement.request_id, old_epoch.request_id)
        self.assertEqual(replacement.round_id, old_epoch.round_id)
        self.assertGreater(replacement.version, old_epoch.version)
        self.assertEqual(ledger.request("r").active_epoch, replacement)
        self.assertEqual(ledger.request("r").outcomes, ())
        self.assertTrue(all(not ledger.allocator.is_live(page.handle) for page in old_branch.pages))
        stale_operations = [
            lambda: ledger.stage_outcome(old_epoch, "late", 1),
            lambda: ledger.commit(old_epoch, "old", 1),
            lambda: ledger.abort(old_epoch),
            lambda: ledger.retry(old_epoch),
        ]
        for operation in stale_operations:
            with self.subTest(operation=operation), self.assertRaises(StaleEpochError):
                operation()

        ledger.stage_outcome(replacement, "current", 2)
        result = ledger.commit(replacement, "current", 2)
        self.assertEqual(result.committed_blocks, 5)
        ledger.audit()

    def test_invalid_selection_and_prefix_leave_transaction_untouched(self) -> None:
        ledger = self.make_ledger()
        ledger.register_request("r", 4)
        epoch = ledger.begin("r", 0)
        branch = ledger.stage_outcome(epoch, "known", 2)
        with self.assertRaises(OutcomeNotFoundError):
            ledger.commit(epoch, "unknown", 1)
        with self.assertRaises(InvalidCommitError):
            ledger.commit(epoch, "known", 3)
        with self.assertRaises(DuplicateOutcomeError):
            ledger.stage_outcome(epoch, "known", 2)
        self.assertTrue(all(ledger.allocator.is_live(page.handle) for page in branch.pages))
        self.assertEqual(ledger.request("r").active_epoch, epoch)
        ledger.audit()

    def test_out_of_pages_is_atomic_for_registration_and_staging(self) -> None:
        ledger = self.make_ledger(pages=3)
        ledger.register_request("r", 8)
        with self.assertRaises(OutOfPagesError):
            ledger.register_request("too-large", 8)
        self.assertEqual(len(ledger.requests()), 1)
        self.assertEqual(ledger.allocator.free_count, 1)

        epoch = ledger.begin("r", 0)
        with self.assertRaises(OutOfPagesError):
            ledger.stage_outcome(epoch, "needs-two", 8)
        self.assertEqual(ledger.request("r").outcomes, ())
        self.assertEqual(ledger.allocator.free_count, 1)
        ledger.abort(epoch)
        ledger.audit()

    def test_registration_rolls_back_if_allocator_fails_after_partial_progress(self) -> None:
        class FaultAllocator(FixedPageAllocator):
            def __init__(self) -> None:
                super().__init__(4, 4)
                self.calls = 0

            def allocate(self, owner: str | None = None) -> PageHandle:
                self.calls += 1
                if self.calls == 2:
                    raise OutOfPagesError("injected allocation fault")
                return super().allocate(owner)

        allocator = FaultAllocator()
        ledger = SpeculativeLedger(allocator)
        with self.assertRaises(OutOfPagesError):
            ledger.register_request("r", 8)
        self.assertEqual(allocator.allocated_count, 0)
        self.assertEqual(allocator.free_count, allocator.page_count)
        self.assertEqual(ledger.requests(), ())
        ledger.audit()

    def test_request_lifecycle_and_exclusive_allocator_audit(self) -> None:
        ledger = self.make_ledger()
        ledger.register_request(1, 5)
        with self.assertRaises(DuplicateRequestError):
            ledger.register_request(1)
        leaked = ledger.allocator.allocate("outside-ledger")
        with self.assertRaises(InvariantViolation):
            ledger.audit()
        ledger.allocator.release(leaked)
        ledger.audit()
        self.assertTrue(ledger.drop_request(1))
        self.assertFalse(ledger.drop_request(1))
        self.assertEqual(ledger.allocator.allocated_count, 0)

        nonempty = FixedPageAllocator(2, 4)
        nonempty.allocate()
        with self.assertRaises(InvalidConfigurationError):
            SpeculativeLedger(nonempty)

    def test_request_id_reuse_cannot_recreate_an_old_epoch(self) -> None:
        ledger = self.make_ledger()
        ledger.register_request("reused", 2)
        old_epoch = ledger.begin("reused", 0)
        ledger.stage_outcome(old_epoch, "old", 2)
        ledger.drop_request("reused")

        ledger.register_request("reused", 0)
        new_epoch = ledger.begin("reused", 0)
        self.assertGreater(new_epoch.version, old_epoch.version)
        self.assertNotEqual(new_epoch, old_epoch)
        with self.assertRaises(StaleEpochError):
            ledger.stage_outcome(old_epoch, "late-old-reply", 1)
        ledger.stage_outcome(new_epoch, "current", 1)
        ledger.abort(new_epoch)

        # Tombstone counters are part of recovery state even with no live
        # request, so a crash cannot roll the incarnation fence backward.
        ledger.drop_request("reused")
        restored = SpeculativeLedger.from_snapshot(ledger.snapshot())
        restored.register_request("reused", 0)
        after_crash = restored.begin("reused", 0)
        self.assertGreater(after_crash.version, new_epoch.version)
        with self.assertRaises(StaleEpochError):
            restored.stage_outcome(new_epoch, "late-after-crash", 1)
        restored.abort(after_crash)
        restored.audit()

    def test_snapshot_restores_active_cow_transaction_and_abort_replay(self) -> None:
        ledger = self.make_ledger()
        ledger.register_request("active", 7)
        active_epoch = ledger.begin("active", 3)
        ledger.stage_outcome(active_epoch, "a", 6)
        ledger.stage_outcome(active_epoch, "b", 2)

        ledger.register_request("aborted", 2)
        aborted_epoch = ledger.begin("aborted", 0)
        ledger.stage_outcome(aborted_epoch, "x", 3)
        ledger.abort(aborted_epoch)

        snapshot = ledger.snapshot()
        self.assertEqual(ledger.snapshot(), snapshot)
        restored = SpeculativeLedger.from_snapshot(snapshot)
        restored_snapshot = restored.snapshot()
        self.assertEqual(restored.snapshot(), restored_snapshot)
        self.assertNotEqual(restored.allocator.pool_id, ledger.allocator.pool_id)
        self.assertEqual(restored_snapshot.aborted_epochs, snapshot.aborted_epochs)
        self.assertEqual(restored_snapshot.version_floors, snapshot.version_floors)
        self.assertEqual(
            restored_snapshot.allocator.generations,
            snapshot.allocator.generations,
        )
        self.assertEqual(
            restored_snapshot.allocator.free_pages,
            snapshot.allocator.free_pages,
        )
        original_live_handle = snapshot.allocator.live[0][0]
        with self.assertRaises(ForeignPageHandleError):
            restored.allocator.is_live(original_live_handle)
        self.assertTrue(
            all(
                page.handle.pool_id == restored.allocator.pool_id
                for request_view in restored.requests()
                for page in (
                    *request_view.committed_pages,
                    *(
                        branch_page
                        for branch_view in request_view.outcomes
                        for branch_page in branch_view.pages
                    ),
                )
            )
        )
        self.assertFalse(restored.abort(aborted_epoch))
        result = restored.commit(active_epoch, "a", 5)
        self.assertEqual(result.committed_blocks, 12)
        restored.audit()

        # The classmethod alias is also the documented recovery constructor.
        recovered = SpeculativeLedger.recover(restored.snapshot())
        recovered.audit()
        self.assertEqual(recovered.request("active").committed_blocks, 12)

    def test_corrupt_ledger_snapshot_with_duplicate_handle_is_rejected(self) -> None:
        ledger = self.make_ledger()
        ledger.register_request("r", 5)
        epoch = ledger.begin("r", 0)
        ledger.stage_outcome(epoch, "x", 2)
        snapshot = ledger.snapshot()
        request = snapshot.requests[0]
        assert request.active is not None
        branch = request.active.branches[0]
        duplicate_span = PageSpan(
            request.committed_pages[0].handle,
            branch.pages[0].start,
            branch.pages[0].used,
        )
        corrupt_branch = replace(branch, pages=(duplicate_span,))
        corrupt_active = replace(request.active, branches=(corrupt_branch,))
        corrupt_request = replace(request, active=corrupt_active)
        corrupted = replace(snapshot, requests=(corrupt_request,))
        with self.assertRaises(SnapshotError):
            SpeculativeLedger.from_snapshot(corrupted)

        foreign_allocator = FixedPageAllocator(
            snapshot.allocator.page_count, snapshot.allocator.page_size
        )
        foreign_span = PageSpan(
            PageHandle(
                branch.pages[0].handle.page_id,
                branch.pages[0].handle.generation,
                foreign_allocator.pool_id,
            ),
            branch.pages[0].start,
            branch.pages[0].used,
        )
        foreign_branch = replace(branch, pages=(foreign_span,))
        foreign_active = replace(request.active, branches=(foreign_branch,))
        foreign_request = replace(request, active=foreign_active)
        with self.assertRaises(SnapshotError):
            SpeculativeLedger.from_snapshot(replace(snapshot, requests=(foreign_request,)))

        wrong_owner_live = tuple(
            (handle, "wrong-owner") for handle, _owner in snapshot.allocator.live
        )
        wrong_owner_allocator = replace(snapshot.allocator, live=wrong_owner_live)
        with self.assertRaises(SnapshotError):
            SpeculativeLedger.from_snapshot(replace(snapshot, allocator=wrong_owner_allocator))

        invalid_counters = replace(request, version=True)
        with self.assertRaises(SnapshotError):
            SpeculativeLedger.from_snapshot(replace(snapshot, requests=(invalid_counters,)))

    def test_randomized_transactions_preserve_model_and_audit_every_step(self) -> None:
        rng = random.Random(0xF15510)
        ledger = self.make_ledger(pages=256, page_size=8)
        request_ids = [f"req-{index}" for index in range(8)]
        model: dict[str, int] = {}
        next_round: dict[str, int] = {}
        branch_model: dict[str, dict[str, int] | None] = {}
        outcome_counter = 0

        for request_id in request_ids:
            initial = rng.randrange(0, 17)
            ledger.register_request(request_id, initial)
            model[request_id] = initial
            next_round[request_id] = 0
            branch_model[request_id] = None

        for _ in range(750):
            request_id = rng.choice(request_ids)
            active = branch_model[request_id]
            if active is None:
                if rng.random() < 0.08:
                    ledger.drop_request(request_id)
                    initial = rng.randrange(0, 17)
                    ledger.register_request(request_id, initial)
                    model[request_id] = initial
                    branch_model[request_id] = None
                else:
                    ledger.begin(request_id, next_round[request_id])
                    next_round[request_id] += 1
                    branch_model[request_id] = {}
            else:
                view = ledger.request(request_id)
                assert view.active_epoch is not None
                if not active or (len(active) < 4 and rng.random() < 0.62):
                    outcome_id = f"outcome-{outcome_counter}"
                    outcome_counter += 1
                    appended = rng.randrange(0, 19)
                    try:
                        ledger.stage_outcome(view.active_epoch, outcome_id, appended)
                    except OutOfPagesError:
                        ledger.abort(view.active_epoch)
                        branch_model[request_id] = None
                    else:
                        active[outcome_id] = appended
                elif rng.random() < 0.78:
                    outcome_id = rng.choice(list(active))
                    accepted = rng.randrange(active[outcome_id] + 1)
                    ledger.commit(view.active_epoch, outcome_id, accepted)
                    model[request_id] += accepted
                    branch_model[request_id] = None
                else:
                    ledger.abort(view.active_epoch)
                    branch_model[request_id] = None

            ledger.audit()
            for checked_id in request_ids:
                checked = ledger.request(checked_id)
                self.assertEqual(checked.committed_blocks, model[checked_id])
                self.assert_aligned(ledger, checked_id)

        for request_id in request_ids:
            ledger.drop_request(request_id)
        ledger.audit()
        self.assertEqual(ledger.allocator.allocated_count, 0)
        self.assertEqual(ledger.allocator.free_count, ledger.allocator.page_count)


if __name__ == "__main__":
    unittest.main()
