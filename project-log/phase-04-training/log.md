# Phase 04 - SFT, DPO, and Experiment Matrix

## Goal

Run local dry runs and cloud GPU training for the selected model matrix, then evaluate each result with the frozen evaluation set.

## Deliverables

- Local dry-run records
- SFT LoRA / QLoRA adapters
- DPO adapters
- Training configs
- M1 / M2 evaluation reports

## Work Log

| Date | Task | Result | Notes |
|---|---|---|---|

## Decisions

- Use LLaMA-Factory for standard SFT and DPO.
- Use local dry runs before cloud GPU training.

## Artifacts

- TBD

## Open Issues

- Finalize first model matrix.
- Pin LLaMA-Factory, transformers, peft, and trl versions.

## Next Steps

- Create initial SFT and DPO YAML templates.
- Add dry-run command script.

