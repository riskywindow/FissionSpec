# Frozen public replay inputs

`azure_llm_code_v3_1024.csv` is a bounded derivative of Microsoft Azure and
Microsoft Research's **Azure LLM Inference Trace 2024, code service**. The
upstream dataset describes production request arrival timestamps plus context
and generated-token counts; it does not expose prompt content.

- Source documentation:
  <https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md>
- Exact release asset:
  <https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv>
- Upstream license: [CC BY 4.0](https://github.com/Azure/AzurePublicDataset/blob/master/LICENSE)
- Required publication citation: Jovan Stojkovic, Chaojie Zhang, Íñigo Goiri,
  Josep Torrellas, and Esha Choukse, “DynamoLLM: Designing LLM Inference
  Clusters for Performance and Energy Efficiency,” HPCA 2025.

FissionSpec modifies the source by selecting 1,024 consecutive requests from a
start time determined solely by the raw file's SHA-256 digest, converting
absolute timestamps to relative milliseconds, renaming the token-count fields,
and adding deterministic request IDs, a validation label, and source-row
provenance. The complete transformation and hashes are recorded in the adjacent
manifest.

Recreate the derivative after obtaining the 692 MB raw asset:

```bash
PYTHONPATH=src python tools/freeze_public_replay.py freeze \
  /path/to/AzureLLMInferenceTrace_code_1week.csv --force
```

Verify the checked-in derivative without downloading the raw source:

```bash
PYTHONPATH=src python tools/freeze_public_replay.py verify
```

The replay is registered workload input, not model output or performance
evidence. Stage 1 calibration determines only the scalar arrival-time rescaling
needed for the preregistered 0.70 offered-load anchor; request ordering and
shapes remain frozen.
