# Phase 03 - SFT and DPO Dataset Construction

## Goal

Create training data for SFT and DPO after baseline weaknesses are known.

## Deliverables

- SFT `train.jsonl` / `val.jsonl`
- DPO preference pairs
- LLaMA-Factory `dataset_info.json`
- Dataset validation report
- Dataset version record

## Work Log

| Date | Task | Result | Notes |
|---|---|---|---|

## Decisions

- Training data must not contain evaluation-only fields.
- Training and evaluation samples must have zero overlap.

## Artifacts

- TBD

## Open Issues

- Decide first version sample count and category ratio.
- Define synthetic generation and manual audit workflow.

## Next Steps

- Implement dataset validators.
- Draft sample templates for intermediate process and final JSON cases.

