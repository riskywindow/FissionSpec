"""No-download neural micro-model semantic smoke tests."""

from __future__ import annotations

import unittest

from fissionspec.micro_model import (
    MicroModelConfig,
    RandomRecurrentMicroModel,
    run_micro_model_smoke,
)


class RandomRecurrentMicroModelTests(unittest.TestCase):
    def test_initialization_and_exact_table_are_reproducible(self) -> None:
        config = MicroModelConfig(
            seed="same-seed",
            vocab_size=3,
            hidden_size=4,
            context_window=2,
            probability_resolution=4096,
        )
        first = RandomRecurrentMicroModel(config)
        second = RandomRecurrentMicroModel(config)
        self.assertEqual(first, second)
        self.assertEqual(first.to_exact_model(), second.to_exact_model())
        self.assertNotEqual(
            first.to_exact_model().fingerprint,
            RandomRecurrentMicroModel(
                MicroModelConfig(
                    seed="different-seed",
                    vocab_size=3,
                    hidden_size=4,
                    context_window=2,
                    probability_resolution=4096,
                )
            )
            .to_exact_model()
            .fingerprint,
        )

    def test_context_logits_softmax_and_quantization_are_well_formed(self) -> None:
        config = MicroModelConfig(
            seed=7,
            vocab_size=4,
            hidden_size=3,
            context_window=2,
            probability_resolution=10_000,
        )
        model = RandomRecurrentMicroModel(config)
        probabilities = model.probabilities((0, 1, 2))
        weights = model.exact_weights((0, 1, 2))
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertEqual(sum(weights), config.probability_resolution)
        self.assertGreater(min(weights), 0)
        self.assertEqual(model.logits((1, 2)), model.logits((0, 1, 2)))
        self.assertNotEqual(model.logits((1, 2)), model.logits((2, 1)))

        maximum_error = max(
            abs(probability - weight / config.probability_resolution)
            for probability, weight in zip(probabilities, weights, strict=True)
        )
        self.assertLessEqual(maximum_error, config.vocab_size / config.probability_resolution)

    def test_full_semantic_smoke_has_fixed_case_count_and_warning(self) -> None:
        result = run_micro_model_smoke()
        self.assertEqual(result.target_context_rows, 13)
        self.assertEqual(result.draft_context_rows, 13)
        self.assertEqual(result.exact_distribution_cases, 36)
        self.assertEqual(result.greedy_cases, 36)
        self.assertNotEqual(result.target_fingerprint, result.draft_fingerprint)
        self.assertIn("not a real-model or GPU result", result.evidence_warning)
        self.assertEqual(result, run_micro_model_smoke())

    def test_invalid_shapes_and_prefixes_fail_loudly(self) -> None:
        for kwargs in (
            {"vocab_size": 1},
            {"hidden_size": 0},
            {"context_window": 0},
            {"vocab_size": 4, "probability_resolution": 3},
            {"parameter_scale": 0.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MicroModelConfig(seed="invalid", **kwargs)
        with self.assertRaises(ValueError):
            RandomRecurrentMicroModel(MicroModelConfig(seed="valid")).logits((99,))
        with self.assertRaises(TypeError):
            RandomRecurrentMicroModel("not-a-config")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
