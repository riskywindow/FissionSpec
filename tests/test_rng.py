"""Adversarial tests for schedule-independent counter-addressed randomness."""

from __future__ import annotations

import math
import random
import unittest
from concurrent.futures import ThreadPoolExecutor

from fissionspec.rng import (
    CounterRNG,
    InvalidRNGKey,
    RNGError,
    RNGKey,
    bernoulli,
    counter_u64,
    uniform,
)


class CounterRNGTests(unittest.TestCase):
    def test_known_vectors_lock_cross_process_encoding(self) -> None:
        rng = CounterRNG("paper-seed")
        vectors = {
            ("req-7", 0, "accept", 0): 16002291407207737047,
            ("req-7", 0, "accept", 1): 8000155126905624719,
            ("req-7", 1, "accept", 0): 17677315970552524261,
            (7, 0, b"accept", 0): 7961149819416687564,
        }
        for address, expected in vectors.items():
            with self.subTest(address=address):
                self.assertEqual(rng.uint64(*address), expected)

    def test_default_draw_is_zero(self) -> None:
        rng = CounterRNG(91)
        self.assertEqual(rng.uint64("r", 2, "s"), rng.uint64("r", 2, "s", 0))
        self.assertEqual(rng.uniform("r", 2, "s"), rng.uniform("r", 2, "s", 0))
        self.assertEqual(
            rng.bernoulli(0.5, "r", 2, "s"),
            rng.bernoulli(0.5, "r", 2, "s", 0),
        )

    def test_schedule_and_thread_interleaving_do_not_change_values(self) -> None:
        rng = CounterRNG(b"parallel-seed")
        addresses = [
            (f"request-{request}", round_id, stream, draw)
            for request in range(19)
            for round_id in range(5)
            for stream in ("accept", "jitter", "routing")
            for draw in range(4)
        ]
        baseline = {address: rng.uint64(*address) for address in addresses}

        shuffled = list(addresses)
        random.Random(20260722).shuffle(shuffled)
        serial_reordered = {address: rng.uint64(*address) for address in shuffled}
        self.assertEqual(serial_reordered, baseline)

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda address: rng.uint64(*address), shuffled))
        threaded = dict(zip(shuffled, results, strict=True))
        self.assertEqual(threaded, baseline)

    def test_every_address_dimension_is_domain_separated(self) -> None:
        base = RNGKey("seed", "request", 3, "stream", 4)
        variants = [
            base,
            RNGKey("other-seed", "request", 3, "stream", 4),
            RNGKey("seed", "other-request", 3, "stream", 4),
            RNGKey("seed", "request", 2, "stream", 4),
            RNGKey("seed", "request", 3, "other-stream", 4),
            RNGKey("seed", "request", 3, "stream", 5),
        ]
        values = {key.uint64() for key in variants}
        self.assertEqual(len(values), len(variants))

    def test_typed_framing_avoids_common_concatenation_collisions(self) -> None:
        rng = CounterRNG("seed")
        values = {
            rng.uint64(1, 0, "23", 4),
            rng.uint64("1", 0, "23", 4),
            rng.uint64(b"1", 0, "23", 4),
            rng.uint64("12", 0, "3", 4),
            rng.uint64("1", 0, "234", 0),
        }
        self.assertEqual(len(values), 5)

    def test_uniform_is_reproducible_and_half_open(self) -> None:
        rng = CounterRNG("range")
        values = [
            rng.uniform("req", round_id, "u", draw) for round_id in range(20) for draw in range(100)
        ]
        self.assertTrue(all(math.isfinite(value) and 0.0 <= value < 1.0 for value in values))
        self.assertGreater(len(set(values)), 1990)
        self.assertEqual(
            values,
            [
                rng.uniform("req", round_id, "u", draw)
                for round_id in range(20)
                for draw in range(100)
            ],
        )

    def test_provenance_is_stable_and_seed_separated(self) -> None:
        self.assertEqual(CounterRNG("seed").provenance, CounterRNG("seed").provenance)
        self.assertNotEqual(CounterRNG("seed").provenance, CounterRNG("other").provenance)
        self.assertTrue(CounterRNG("seed").provenance.startswith("fissionspec-counter-rng-v1:"))

    def test_bernoulli_boundaries_and_threshold(self) -> None:
        rng = CounterRNG(123)
        for draw in range(100):
            self.assertFalse(rng.bernoulli(0.0, "r", 0, "b", draw))
            self.assertTrue(rng.bernoulli(1.0, "r", 0, "b", draw))
            value = rng.uniform("r", 0, "b", draw)
            self.assertEqual(rng.bernoulli(0.37, "r", 0, "b", draw), value < 0.37)

    def test_functional_and_key_apis_match_object_api(self) -> None:
        key = RNGKey("seed", 8, 9, b"stream", 10)
        rng = CounterRNG("seed")
        self.assertEqual(key.uint64(), counter_u64("seed", 8, 9, b"stream", 10))
        self.assertEqual(key.uniform(), uniform("seed", 8, 9, b"stream", 10))
        self.assertEqual(key.bernoulli(0.2), bernoulli(0.2, "seed", 8, 9, b"stream", 10))
        self.assertEqual(key.uint64(), rng.uint64(8, 9, b"stream", 10))

    def test_invalid_keys_fail_loudly_even_at_bernoulli_boundaries(self) -> None:
        rng = CounterRNG("seed")
        invalid_calls = [
            lambda: CounterRNG(True),
            lambda: rng.uint64("r", -1, "s"),
            lambda: rng.uint64("r", 0, "s", -1),
            lambda: rng.uint64(True, 0, "s"),
            lambda: rng.uint64("r", 0, object()),
            lambda: rng.bernoulli(0.0, "r", -1, "s"),
            lambda: RNGKey("seed", "r", 0, "s", -2),
        ]
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(InvalidRNGKey):
                call()

    def test_invalid_probabilities_are_rejected(self) -> None:
        rng = CounterRNG("seed")
        for probability in (-0.1, 1.1, math.nan, math.inf, -math.inf, True, "0.5"):
            with self.subTest(probability=probability), self.assertRaises(RNGError):
                rng.bernoulli(probability, "r", 0, "s")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
