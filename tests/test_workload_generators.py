from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fissionspec.rng import CounterRNG
from fissionspec.workload_generators import (
    ArrivalTrace,
    load_trace_csv,
    mmpp_arrivals,
    pareto_arrivals,
    poisson_arrivals,
    workload_from_arrivals,
)


class ArrivalGeneratorTests(unittest.TestCase):
    def test_poisson_is_reproducible_and_provenanced(self) -> None:
        left = poisson_arrivals(
            count=100,
            mean_interarrival_ms=2.5,
            rng=CounterRNG("arrival-seed"),
        )
        right = poisson_arrivals(
            count=100,
            mean_interarrival_ms=2.5,
            rng=CounterRNG("arrival-seed"),
        )
        self.assertEqual(left, right)
        self.assertEqual(left.sha256, right.sha256)
        self.assertEqual(len(left.times_ms), 100)
        self.assertEqual(left.times_ms[0], 0.0)
        self.assertEqual(tuple(sorted(left.times_ms)), left.times_ms)
        self.assertNotEqual(
            left.sha256,
            poisson_arrivals(
                count=100,
                mean_interarrival_ms=2.5,
                rng=CounterRNG("other-seed"),
            ).sha256,
        )

    def test_pareto_has_minimum_gap_and_heavy_tail(self) -> None:
        trace = pareto_arrivals(
            count=300,
            minimum_interarrival_ms=1.25,
            tail_index=1.4,
            rng=CounterRNG(9),
        )
        gaps = tuple(
            right - left for left, right in zip(trace.times_ms, trace.times_ms[1:], strict=False)
        )
        self.assertTrue(all(gap >= 1.25 for gap in gaps))
        self.assertGreater(max(gaps), 10 * min(gaps))
        with self.assertRaises(ValueError):
            pareto_arrivals(
                count=2,
                minimum_interarrival_ms=1,
                tail_index=1,
                rng=CounterRNG(1),
            )

    def test_mmpp_exact_race_reproducibly_visits_bursty_states(self) -> None:
        trace = mmpp_arrivals(
            count=500,
            arrival_rates_per_ms=(0.05, 3.0),
            transition_rates_per_ms=(0.02, 0.2),
            rng=CounterRNG("mmpp"),
        )
        again = mmpp_arrivals(
            count=500,
            arrival_rates_per_ms=(0.05, 3.0),
            transition_rates_per_ms=(0.02, 0.2),
            rng=CounterRNG("mmpp"),
        )
        self.assertEqual(trace, again)
        gaps = tuple(
            right - left for left, right in zip(trace.times_ms, trace.times_ms[1:], strict=False)
        )
        self.assertLess(min(gaps), 0.01)
        self.assertGreater(max(gaps), 10.0)
        self.assertEqual(trace.process, "mmpp-2state")

    def test_arrivals_materialize_without_losing_shape(self) -> None:
        arrivals = poisson_arrivals(
            count=8,
            mean_interarrival_ms=1,
            rng=CounterRNG(1),
        )
        workload = workload_from_arrivals(
            arrivals,
            name="paired-poisson",
            prompt_tokens=128,
            output_tokens=17,
            cache_hit_probability=(0.9, 0.7),
        )
        self.assertEqual(tuple(row.arrival_ms for row in workload), arrivals.times_ms)
        self.assertTrue(all(row.prompt_tokens == 128 for row in workload))
        self.assertTrue(all(row.output_tokens == 17 for row in workload))

    def test_arrival_trace_rejects_unsorted_times(self) -> None:
        with self.assertRaises(ValueError):
            ArrivalTrace(
                (1.0, 0.0),
                "bad",
                "rng",
                (),
            )


class TraceReplayTests(unittest.TestCase):
    def test_checked_in_example_has_disjoint_train_and_validation_ids(self) -> None:
        path = Path(__file__).resolve().parents[1] / "configs" / "replay_trace.example.csv"
        train = load_trace_csv(path, split="train")
        validation = load_trace_csv(path, split="validation")
        train_ids = {row.request_id for row in train.workload}
        validation_ids = {row.request_id for row in validation.workload}
        self.assertTrue(train_ids)
        self.assertTrue(validation_ids)
        self.assertTrue(train_ids.isdisjoint(validation_ids))
        self.assertEqual(train.source_sha256, validation.source_sha256)

    def test_csv_replay_filters_explicit_splits_and_hashes_source(self) -> None:
        contents = (
            "request_id,arrival_ms,output_tokens,split,prompt_tokens,"
            "speculation_length,cache_hit_probability,"
            "token_acceptance_probability,tbt_slo_ms,deadline_ms,tenant\n"
            "a,0,8,train,128,4,0.9;0.7,0.8,20,200,alpha\n"
            "b,1.5,16,validation,256,6,0.5,0.6;0.4,25,,beta\n"
            "c,2,4,train,32,2,,,30,,gamma\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "serving.csv"
            path.write_text(contents, encoding="utf-8")
            train = load_trace_csv(path, split="train")
            validation = load_trace_csv(path, split="validation")

        self.assertEqual(train.source_sha256, validation.source_sha256)
        self.assertEqual(train.source_rows, 3)
        self.assertEqual(train.selected_rows, 2)
        self.assertEqual(tuple(row.request_id for row in train.workload), ("a", "c"))
        self.assertEqual(
            train.workload.requests[0].cache_hit_probability,
            (0.9, 0.7),
        )
        self.assertEqual(validation.workload.requests[0].deadline_ms, None)
        self.assertEqual(validation.workload.requests[0].prompt_tokens, 256)

    def test_csv_replay_rejects_missing_or_empty_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.csv"
            missing.write_text("request_id,arrival_ms\nr,0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_trace_csv(missing)

            split = Path(directory) / "split.csv"
            split.write_text(
                "request_id,arrival_ms,output_tokens,split\nr,0,2,train\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_trace_csv(split, split="validation")


if __name__ == "__main__":
    unittest.main()
