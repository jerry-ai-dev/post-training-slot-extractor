# Phase 02 - Evaluation Dataset and M0 Baseline

## Goal

Create and freeze the evaluation dataset, then establish the M0 baseline scorecard.

## Deliverables

- Frozen `test.jsonl`
- Evaluation data validation report
- M0 baseline scorecard
- Baseline run records under `experiments/baselines/`

## Work Log

| Date | Task | Result | Notes |
|---|---|---|---|

## Decisions

- Evaluation data must stay isolated from training data.
- Frozen evaluation data should not be edited after M0 unless a new version is explicitly created.

## Artifacts

- TBD

## Open Issues

- Decide whether to add a hidden holdout set beyond `test.jsonl`.
- Define exact case tags and assertion types.

## Next Steps

- Draft first evaluation sample schema.
- Add validation checks for `expected`, `assertions`, `gold_facts`, and `tags`.

