# vLLM integration notes

vLLM is a second target after validating the mechanism in SGLang. The same
four events are sufficient—reserve, target completion, recovery completion,
and next-batch selection—but the implementation must land where scheduler
metadata and KV block tables are assembled, before model-runner inputs become
immutable.

The engine needs an SSD outcome-cache producer first. FissionSpec is not a
replacement for that producer: it consumes per-request hit/miss outcomes and
recovery ETAs. Until vLLM exposes those events, an adapter would be a simulator
or trace replayer, not a production implementation.

See the [SGLang contract](../sglang/README.md) for version fences, physical-work
instrumentation, liveness rules, and acceptance gates. Those obligations are
engine independent.
