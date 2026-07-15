# Phase 05 - Quantization and Local Deployment

## Goal

Merge the selected adapter, convert to GGUF, quantize for local CPU inference, and validate quality and latency after compression.

## Deliverables

- Merged fp16 / bf16 model
- GGUF model
- imatrix calibration output
- Q4_K_M model, plus optional Q5_K_M / Q8_0 comparisons
- llama.cpp local serving record
- Quantization comparison scorecard

## Work Log

| Date | Task | Result | Notes |
|---|---|---|---|

## Decisions

- llama.cpp is the main local CPU inference path.
- Ollama is only a convenience fallback for quick manual checks.

## Artifacts

- TBD

## Open Issues

- Define acceptable quality regression after quantization.
- Define local latency measurement protocol.

## Next Steps

- Add merge and GGUF conversion scripts.
- Add llama-server launch config template.

