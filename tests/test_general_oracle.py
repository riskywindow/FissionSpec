"""Exact, cross-model, and certificate tests for the generalized oracle."""

from __future__ import annotations

import unittest
from dataclasses import replace
from fractions import Fraction

from fissionspec.general_oracle import (
    CertificateVerificationError,
    DispatchEvent,
    ExactLatencySurface,
    GeneralOracleCertificate,
    GeneralOracleError,
    GeneralOracleLimitExceeded,
    GeneralOracleLimitKind,
    MissingLatencyShapeError,
    OracleCapacity,
    OracleJob,
    OracleObjective,
    OracleProblem,
    OracleSearchLimits,
    OracleWaitConfig,
    WaitEvent,
    WaitKind,
    objective_gap,
    score_completion_times,
    solve_general_oracle,
    verify_general_oracle_certificate,
    work_conserving_edf,
)
from fissionspec.oracle import OracleAction, offline_coalescing_oracle
from fissionspec.policies import FissionSpecPolicy
from fissionspec.profiles import HardwareProfile, LatencyCurve
from fissionspec.rng import CounterRNG
from fissionspec.simulator import simulate
from fissionspec.workload import RequestConfig, Workload

F = Fraction
ONE_FIFTH = F(1, 5)
FIVE_FOURTHS = F(5, 4)


def _limits(
    *,
    jobs: int = 8,
    states: int = 20_000,
    transitions: int = 200_000,
    trace_events: int = 200,
) -> OracleSearchLimits:
    return OracleSearchLimits(jobs, states, transitions, trace_events)


def _two_job_problem(
    *,
    second_release: Fraction = ONE_FIFTH,
    batch_duration: Fraction = FIVE_FOURTHS,
    wait: OracleWaitConfig | None = None,
) -> OracleProblem:
    return OracleProblem(
        (
            OracleJob("first", 0, 1, 50),
            OracleJob("second", second_release, 1, 50),
        ),
        OracleCapacity(2, 2),
        ExactLatencySurface({(1, 1): 1, (2, 2): batch_duration}),
        wait=(
            OracleWaitConfig(
                include_release_times=True,
                include_deadline_safe_times=False,
                latest_optional_time=2,
            )
            if wait is None
            else wait
        ),
    )


class ExactSearchTests(unittest.TestCase):
    def test_arbitrary_subset_avoids_pathological_full_batch(self) -> None:
        problem = OracleProblem(
            (
                OracleJob("urgent", 0, 1, 2),
                OracleJob("lax", 0, 1, 100),
            ),
            OracleCapacity(2, 2),
            ExactLatencySurface({(1, 1): 1, (2, 2): 10}),
            wait=OracleWaitConfig(False, False),
        )
        solved = solve_general_oracle(problem, limits=_limits())
        heuristic = work_conserving_edf(problem)

        self.assertEqual(solved.objective, OracleObjective(0, 3))
        self.assertEqual(heuristic.objective, OracleObjective(1, 20))
        self.assertEqual(
            objective_gap(heuristic.objective, solved.objective).deadline_violation_gap,
            1,
        )
        self.assertEqual(
            objective_gap(heuristic.objective, solved.objective).weighted_flow_gap,
            17,
        )
        self.assertTrue(
            all(isinstance(event, DispatchEvent) and event.rows == 1 for event in solved.events)
        )

    def test_deadline_violations_precede_weighted_flow_lexicographically(self) -> None:
        problem = OracleProblem(
            (
                OracleJob("urgent", 0, 1, 1, weight=1),
                OracleJob("heavy", 0, 1, 100, weight=100),
            ),
            OracleCapacity(1, 1),
            ExactLatencySurface({(1, 1): 1}),
            wait=OracleWaitConfig(False, False),
        )
        solved = solve_general_oracle(problem, limits=_limits())

        # Heavy-first has much lower weighted flow (102) but violates the
        # urgent deadline. Urgent-first has zero violations and weighted flow
        # 1 + 100*2 = 201, so it wins the declared lexicographic objective.
        self.assertEqual(solved.objective, OracleObjective(0, 201))
        first = solved.events[0]
        self.assertIsInstance(first, DispatchEvent)
        assert isinstance(first, DispatchEvent)
        self.assertEqual(first.job_ids, ("urgent",))

    def test_two_dimensional_latency_and_slot_capacity_are_binding(self) -> None:
        problem = OracleProblem(
            (
                OracleJob("narrow", 0, 1, 100),
                OracleJob("wide", 0, 2, 100),
            ),
            OracleCapacity(2, 3),
            ExactLatencySurface(
                {
                    (1, 1): 1,
                    (1, 2): 2,
                    (2, 3): F(5, 2),
                }
            ),
            wait=OracleWaitConfig(False, False),
        )
        solved = solve_general_oracle(problem, limits=_limits())

        self.assertEqual(solved.objective, OracleObjective(0, 4))
        dispatches = tuple(event for event in solved.events if isinstance(event, DispatchEvent))
        self.assertEqual(
            tuple(event.job_ids for event in dispatches),
            (("narrow",), ("wide",)),
        )
        self.assertEqual(tuple(event.slots for event in dispatches), (1, 2))

    def test_release_deadline_safe_and_grid_wait_spaces_are_explicit(self) -> None:
        release_problem = _two_job_problem()
        release_solution = solve_general_oracle(release_problem, limits=_limits())
        self.assertEqual(release_solution.objective, OracleObjective(0, F(27, 10)))
        self.assertEqual(
            release_solution.events[0],
            WaitEvent(0, F(1, 5), WaitKind.RELEASE),
        )

        safe_problem = OracleProblem(
            (
                OracleJob("first", 0, 1, F(29, 20)),
                OracleJob("second", F(1, 5), 1, 50),
            ),
            OracleCapacity(2, 2),
            ExactLatencySurface({(1, 1): 1, (2, 2): F(5, 4)}),
            wait=OracleWaitConfig(
                include_release_times=False,
                include_deadline_safe_times=True,
                latest_optional_time=1,
            ),
        )
        safe_solution = solve_general_oracle(safe_problem, limits=_limits())
        self.assertEqual(
            safe_solution.events[0],
            WaitEvent(0, F(1, 5), WaitKind.DEADLINE_SAFE),
        )

        grid_problem = _two_job_problem(
            wait=OracleWaitConfig(
                include_release_times=False,
                include_deadline_safe_times=False,
                grid_times=(F(1, 5),),
                latest_optional_time=1,
            )
        )
        grid_solution = solve_general_oracle(grid_problem, limits=_limits())
        self.assertEqual(
            grid_solution.events[0],
            WaitEvent(0, F(1, 5), WaitKind.GRID),
        )

        overlap = _two_job_problem(
            wait=OracleWaitConfig(
                include_release_times=True,
                include_deadline_safe_times=True,
                grid_times=(F(1, 5),),
                latest_optional_time=1,
            )
        )
        self.assertEqual(dict(overlap.decision_points)[F(1, 5)], WaitKind.RELEASE)

    def test_memoization_and_order_dominance_are_observable_and_stable(self) -> None:
        problem = OracleProblem(
            (OracleJob("b", 0, 1, 10), OracleJob("a", 0, 1, 10)),
            OracleCapacity(2, 2),
            ExactLatencySurface({(1, 1): 1, (2, 2): 2}),
            wait=OracleWaitConfig(False, False),
        )
        first = solve_general_oracle(problem, limits=_limits())
        second = solve_general_oracle(problem, limits=_limits())

        self.assertEqual(first, second)
        self.assertGreater(first.states_pruned_by_memo, 0)
        self.assertGreater(first.transitions_pruned_by_dominance, 0)
        self.assertGreater(first.states_explored, 0)
        self.assertGreater(first.transitions_explored, 0)
        self.assertEqual(first.events[0].job_ids, ("a",))  # type: ignore[union-attr]


class CertificateTests(unittest.TestCase):
    def test_independent_brute_force_verifies_optimum_and_tie_break(self) -> None:
        problem = _two_job_problem()
        certificate = solve_general_oracle(problem, limits=_limits())
        report = verify_general_oracle_certificate(
            problem,
            certificate,
            max_verifier_nodes=100_000,
        )

        self.assertTrue(report.optimality_checked)
        self.assertGreater(report.verifier_nodes, 0)
        self.assertEqual(report.objective, certificate.objective)
        self.assertEqual(report.completion_times, certificate.completion_times)
        self.assertEqual(len(certificate.certificate_hash), 64)

    def test_exhaustive_smaller_domain_agrees_with_unmemoized_verifier(self) -> None:
        checked = 0
        for second_release in (F(0), F(1, 2)):
            for batch_duration in (F(1), F(3, 2)):
                for second_deadline in (F(3, 2), F(3)):
                    problem = OracleProblem(
                        (
                            OracleJob("a", 0, 1, 2),
                            OracleJob(
                                "b",
                                second_release,
                                1,
                                second_deadline,
                            ),
                        ),
                        OracleCapacity(2, 2),
                        ExactLatencySurface(
                            {
                                (1, 1): 1,
                                (2, 2): batch_duration,
                            }
                        ),
                        wait=OracleWaitConfig(
                            include_release_times=True,
                            include_deadline_safe_times=False,
                            latest_optional_time=2,
                        ),
                    )
                    certificate = solve_general_oracle(problem, limits=_limits())
                    report = verify_general_oracle_certificate(
                        problem,
                        certificate,
                        max_verifier_nodes=100_000,
                    )
                    self.assertEqual(report.objective, certificate.objective)
                    checked += 1
        self.assertEqual(checked, 8)

    def test_replay_rejects_hash_objective_timing_and_tie_tampering(self) -> None:
        problem = OracleProblem(
            (OracleJob("a", 0, 1, 10), OracleJob("b", 0, 1, 10)),
            OracleCapacity(1, 1),
            ExactLatencySurface({(1, 1): 1}),
            wait=OracleWaitConfig(False, False),
        )
        certificate = solve_general_oracle(problem, limits=_limits())

        with self.assertRaisesRegex(CertificateVerificationError, "input_hash"):
            verify_general_oracle_certificate(
                problem,
                replace(certificate, input_hash="0" * 64),
                prove_optimality=False,
            )
        with self.assertRaisesRegex(CertificateVerificationError, "objective"):
            verify_general_oracle_certificate(
                problem,
                replace(certificate, objective=OracleObjective(0, 4)),
                prove_optimality=False,
            )

        first = certificate.events[0]
        assert isinstance(first, DispatchEvent)
        delayed = replace(first, end_time=first.end_time + 1)
        with self.assertRaisesRegex(CertificateVerificationError, "ends at"):
            verify_general_oracle_certificate(
                problem,
                replace(certificate, events=(delayed, *certificate.events[1:])),
                prove_optimality=False,
            )

        alternate_events = (
            DispatchEvent(0, 1, ("b",), 1, 1),
            DispatchEvent(1, 2, ("a",), 1, 1),
        )
        alternate = GeneralOracleCertificate(
            input_hash=problem.input_hash,
            events=alternate_events,
            objective=OracleObjective(0, 3),
            completion_times=(("a", F(2)), ("b", F(1))),
            states_explored=certificate.states_explored,
            states_pruned_by_memo=certificate.states_pruned_by_memo,
            transitions_explored=certificate.transitions_explored,
            transitions_pruned_by_dominance=certificate.transitions_pruned_by_dominance,
        )
        replay_only = verify_general_oracle_certificate(
            problem,
            alternate,
            prove_optimality=False,
        )
        self.assertFalse(replay_only.optimality_checked)
        with self.assertRaisesRegex(CertificateVerificationError, "tie-break"):
            verify_general_oracle_certificate(
                problem,
                alternate,
                max_verifier_nodes=10_000,
            )

    def test_input_hash_and_certificate_are_input_order_invariant(self) -> None:
        jobs = (
            OracleJob("a", 0, 1, 10, cohort_id="c"),
            OracleJob("b", 0, 1, 10, cohort_id="c"),
        )
        first = OracleProblem(
            jobs,
            OracleCapacity(2, 2),
            ExactLatencySurface({(1, 1): 1, (2, 2): F(3, 2)}),
            wait=OracleWaitConfig(False, False),
        )
        second = OracleProblem(
            tuple(reversed(jobs)),
            OracleCapacity(2, 2),
            ExactLatencySurface({(2, 2): F(3, 2), (1, 1): 1}),
            wait=OracleWaitConfig(False, False),
        )
        first_certificate = solve_general_oracle(first, limits=_limits())
        second_certificate = solve_general_oracle(second, limits=_limits())

        self.assertEqual(first.input_hash, second.input_hash)
        self.assertEqual(first_certificate, second_certificate)
        self.assertEqual(
            first_certificate.certificate_hash,
            second_certificate.certificate_hash,
        )


class ExistingOracleAndHeuristicTests(unittest.TestCase):
    @staticmethod
    def _profile(batch_two_ms: float) -> HardwareProfile:
        return HardwareProfile(
            target_curve=LatencyCurve(((1, 1.0), (2, batch_two_ms))),
            draft_curve=LatencyCurve(((1, 0.1), (2, 0.2))),
            recovery_curve=LatencyCurve(((1, 0.2), (2, 0.3))),
            verifier_slot_ms=0.0,
            name="general-oracle-cross-check",
        )

    def test_restricted_oracle_matches_on_binary_common_domain(self) -> None:
        workload = Workload(
            (
                RequestConfig(
                    "first",
                    arrival_ms=0.0,
                    output_tokens=1,
                    speculation_length=1,
                    deadline_ms=50.0,
                ),
                RequestConfig(
                    "second",
                    arrival_ms=0.2,
                    output_tokens=1,
                    speculation_length=1,
                    deadline_ms=50.0,
                ),
            )
        )
        restricted = offline_coalescing_oracle(
            workload,
            self._profile(1.25),
            CounterRNG("common-domain"),
            max_decision_depth=4,
            max_simulations=16,
            max_batch_size=2,
        )
        generalized = solve_general_oracle(_two_job_problem(), limits=_limits())

        self.assertEqual(
            restricted.action_prefix,
            (OracleAction.WAIT_NEXT_READINESS,),
        )
        self.assertAlmostEqual(
            restricted.objective_flow_time_ms,
            float(generalized.objective.weighted_flow),
        )
        self.assertEqual(generalized.objective.weighted_flow, F(27, 10))

    def test_actual_fissionspec_gap_on_adversarial_subset_trace(self) -> None:
        workload = Workload(
            (
                RequestConfig(
                    "urgent",
                    arrival_ms=0.0,
                    output_tokens=1,
                    speculation_length=1,
                    deadline_ms=2.0,
                ),
                RequestConfig(
                    "lax",
                    arrival_ms=0.0,
                    output_tokens=1,
                    speculation_length=1,
                    deadline_ms=100.0,
                ),
            ),
            name="adversarial-full-batch",
        )
        simulated = simulate(
            workload,
            self._profile(10.0),
            FissionSpecPolicy(max_wait_ms=0.0),
            CounterRNG("fissionspec-gap"),
            max_batch_size=2,
        )
        problem = OracleProblem(
            (
                OracleJob("urgent", 0, 1, 2),
                OracleJob("lax", 0, 1, 100),
            ),
            OracleCapacity(2, 2),
            ExactLatencySurface({(1, 1): 1, (2, 2): 10}),
            wait=OracleWaitConfig(False, False),
        )
        optimum = solve_general_oracle(problem, limits=_limits())
        completions = {
            request.request_id: F(str(request.completion_ms)) for request in simulated.requests
        }
        fissionspec_objective = score_completion_times(problem, completions)
        gap = objective_gap(fissionspec_objective, optimum.objective)

        self.assertEqual(optimum.objective, OracleObjective(0, 3))
        self.assertEqual(fissionspec_objective, OracleObjective(1, 20))
        self.assertEqual(gap.deadline_violation_gap, 1)
        self.assertEqual(gap.weighted_flow_gap, 17)
        self.assertEqual(
            tuple(launch.request_ids for launch in simulated.target_launches),
            (("urgent", "lax"),),
        )


class LimitsAndValidationTests(unittest.TestCase):
    def test_exact_inputs_missing_shapes_and_cohort_errors_fail_loudly(self) -> None:
        invalid_calls = [
            lambda: OracleJob("float", 0.1, 1, 2),  # type: ignore[arg-type]
            lambda: OracleJob("bool", 0, True, 2),
            lambda: OracleCapacity(0, 1),
            lambda: ExactLatencySurface({(1, 1): 1.5}),  # type: ignore[dict-item]
            lambda: OracleWaitConfig(include_release_times=1),  # type: ignore[arg-type]
        ]
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(GeneralOracleError):
                call()

        with self.assertRaises(MissingLatencyShapeError):
            OracleProblem(
                (OracleJob("a", 0, 1, 2), OracleJob("b", 0, 1, 2)),
                OracleCapacity(2, 2),
                ExactLatencySurface({(1, 1): 1}),
            )
        with self.assertRaisesRegex(GeneralOracleError, "one release"):
            OracleProblem(
                (
                    OracleJob("a", 0, 1, 2, cohort_id="c"),
                    OracleJob("b", 1, 1, 3, cohort_id="c"),
                ),
                OracleCapacity(1, 1),
                ExactLatencySurface({(1, 1): 1}),
            )

    def test_optimizer_and_independent_verifier_limits_fail_closed(self) -> None:
        problem = _two_job_problem()
        with self.assertRaises(GeneralOracleLimitExceeded) as jobs:
            solve_general_oracle(
                problem,
                limits=OracleSearchLimits(1, 100, 100, 100),
            )
        self.assertEqual(jobs.exception.kind, GeneralOracleLimitKind.JOBS)

        with self.assertRaises(GeneralOracleLimitExceeded) as states:
            solve_general_oracle(
                problem,
                limits=OracleSearchLimits(2, 1, 100, 100),
            )
        self.assertEqual(states.exception.kind, GeneralOracleLimitKind.STATES)

        with self.assertRaises(GeneralOracleLimitExceeded) as transitions:
            solve_general_oracle(
                problem,
                limits=OracleSearchLimits(2, 100, 1, 100),
            )
        self.assertEqual(
            transitions.exception.kind,
            GeneralOracleLimitKind.TRANSITIONS,
        )

        with self.assertRaises(GeneralOracleLimitExceeded) as trace:
            solve_general_oracle(
                problem,
                limits=OracleSearchLimits(2, 100, 100, 1),
            )
        self.assertEqual(trace.exception.kind, GeneralOracleLimitKind.TRACE_EVENTS)

        certificate = solve_general_oracle(problem, limits=_limits())
        with self.assertRaises(GeneralOracleLimitExceeded) as verifier:
            verify_general_oracle_certificate(
                problem,
                certificate,
                max_verifier_nodes=1,
            )
        self.assertEqual(
            verifier.exception.kind,
            GeneralOracleLimitKind.VERIFIER_NODES,
        )

    def test_completion_scoring_requires_exact_complete_mapping(self) -> None:
        problem = _two_job_problem()
        with self.assertRaisesRegex(GeneralOracleError, "every job"):
            score_completion_times(problem, {"first": 1})
        with self.assertRaisesRegex(GeneralOracleError, "int or Fraction"):
            score_completion_times(
                problem,
                {"first": 1.0, "second": 2},  # type: ignore[dict-item]
            )
        with self.assertRaisesRegex(GeneralOracleError, "before its release"):
            score_completion_times(
                problem,
                {"first": 1, "second": 0},
            )


if __name__ == "__main__":
    unittest.main()
