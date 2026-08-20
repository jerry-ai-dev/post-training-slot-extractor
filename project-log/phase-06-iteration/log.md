# Phase 06 - Iteration and Data Loop

## Goal

Use scorecard failures to drive targeted data repair, retraining, and repeated evaluation.

## Deliverables

- Failure analysis records
- Targeted data supplement plan
- Retraining records
- M4 iteration summary

## Work Log

| Date | Task | Result | Notes |
|---|---|---|---|
| 2026-08-20 | Establish iteration scaffold | Added round registry, reusable round template, four report templates, artifact/import/package records, cross-round summaries, and dataset registry | No training or automation implemented |
| 2026-08-20 | Start round-001 problem analysis | Registered the first iteration and documented Phase 05 cross-model failure causes with structured evidence and hypotheses | Strategy and variants intentionally remain unapproved |
| 2026-08-20 | Correct confirmation evaluation | Updated three available-and-confirmed cases to treat user confirmation as booking success and three failed-result acknowledgements to keep confirmation=false; aligned prompt semantics and reply matcher; bumped evaluation contract to v2.4 | Historical Phase 05 reports remain unchanged until re-evaluation |
| 2026-08-20 | Draft round-001 remediation strategy | Recorded an evaluation-first, targeted SFT data redesign plan covering state transitions, tool calls, date normalization, schema stability, and confirmation semantics | Date understanding remains a trained model capability; code is used only to generate and validate labels |
| 2026-08-20 | Confirm SFT-first sequence | Round 001 will fix evaluation, rebuild SFT data, and retrain SFT only; DPO is deferred until residual errors from the new SFT are available | Keeps attribution between SFT and DPO clear |
| 2026-08-20 | Build and freeze sft-v0.2 | Produced 795 unique raw samples, 715 train and 80 validation rows; added 316 targeted cases and removed 21 duplicated v0.1 inputs | Exact eval input overlap is zero; remote configs prepared for 0.6B and 1.7B |

## Decisions

- Use metric and tag-level failures to decide data additions.
- Avoid broad data expansion before confirming the failure category.
- Round 001 prioritizes SFT foundation quality; do not retrain DPO in the same round.
- Build future DPO data from the new SFT model's verified residual hard cases.

## Artifacts

- `experiments/phase06/README.md`
- `experiments/phase06/registry.yaml`
- `experiments/phase06/_template/`
- `experiments/phase06/summary/`
- `data/dataset-registry.yaml`

## Open Issues

- Define failure taxonomy.
- Decide when the iteration curve is flat enough to stop.

## Next Steps

- Create `round-001` from the template after the first optimization strategy is chosen.
- Fill in problem evidence, strategy, variants, and the run matrix before approval.
- Later add validation/export/import commands when automation is needed.
