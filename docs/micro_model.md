# No-download CPU neural smoke model

`fissionspec.micro_model` closes the semantic-oracle integration check without
downloading weights or requiring PyTorch. It is a tiny recurrent neural
language model, initialized deterministically from the counter RNG and
evaluated with ordinary CPU binary64 operations.

For every suffix context up to the configured window, the harness:

1. evaluates recurrent hidden state and next-token logits;
2. applies a stable softmax;
3. quantizes the probabilities to a fixed total integer mass using
   largest-remainder allocation while retaining nonzero support; and
4. constructs an exact rational `TinyAutoregressiveModel`.

Independent target and draft initializations then pass through the same
rejection, residual-correction, target-bonus, greedy, and committed-state
semantics used elsewhere. `run_micro_model_smoke()` checks 36 exact
distribution cases and 36 greedy cases across three prompts, four horizons,
and three speculation widths.

The suffix table is deliberately small enough to enumerate. The neural
floating-point output is frozen into exact rational mass before the proof
oracle runs, so the check exercises a neural-logit-to-speculation seam without
introducing tolerance into the distribution equality.

This is not a transformer, a pretrained model, or evidence about CUDA
numerics. Production acceptance still requires same-revision tokenizer/model
outputs under the actual serving kernels. The smoke model exists to catch
integration mistakes cheaply and reproducibly before that work.
