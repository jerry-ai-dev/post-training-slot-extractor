# src/slot_extractor/evaluation/runner.py
from __future__ import annotations

from slot_extractor.evaluation.scorecard import aggregate_scorecard
from slot_extractor.evaluation.scorer import Scorer
from slot_extractor.evaluation.scorers.field_extraction import FieldExtractionScorer
from slot_extractor.evaluation.scorers.instruction import InstructionScorer
from slot_extractor.evaluation.scorers.not_available import NotAvailableScorer
from slot_extractor.evaluation.scorers.speed import SpeedScorer
from slot_extractor.evaluation.scorers.tool_call import ToolCallScorer
from slot_extractor.inference.base import Backend
from slot_extractor.prompts.template import PromptBuilder
from slot_extractor.schemas.results import CaseResult, Scorecard
from slot_extractor.schemas.sample import Sample


def default_scorers() -> list[Scorer]:
    return [
        InstructionScorer(),
        ToolCallScorer(),
        FieldExtractionScorer(),
        NotAvailableScorer("hallucination"),
        NotAvailableScorer("intent"),
        NotAvailableScorer("restraint"),
        NotAvailableScorer("multiturn"),
        SpeedScorer(),
        NotAvailableScorer("resource"),
    ]


def run_evaluation(
    samples: list[Sample],
    backend: Backend,
    scorers: list[Scorer] | None = None,
) -> Scorecard:
    prompt_builder = PromptBuilder()
    active_scorers = scorers or default_scorers()
    case_results: list[CaseResult] = []
    for sample in samples:
        prompt = prompt_builder.build(sample)
        generation = backend.generate(prompt)
        dimensions = {
            scorer.dimension: scorer.score(sample, generation)
            for scorer in active_scorers
            if scorer.applies_to(sample)
        }
        case_results.append(
            CaseResult(
                sample_id=sample.id,
                layer=sample.layer,
                model_output=generation.text,
                dimensions=dimensions,
            )
        )
    return aggregate_scorecard(model=backend.model, cases=case_results)
