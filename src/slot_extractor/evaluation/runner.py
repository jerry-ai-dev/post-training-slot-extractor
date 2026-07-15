# src/slot_extractor/evaluation/runner.py
from __future__ import annotations

from slot_extractor.evaluation.scorecard import aggregate_scorecard
from slot_extractor.evaluation.scorer import Scorer
from slot_extractor.evaluation.scorers.instruction import InstructionScorer
from slot_extractor.evaluation.scorers.not_available import NotAvailableScorer
from slot_extractor.evaluation.scorers.task_correctness import TaskCorrectnessScorer
from slot_extractor.inference.base import Backend
from slot_extractor.prompts.template import PromptBuilder
from slot_extractor.schemas.results import CaseResult, Scorecard
from slot_extractor.schemas.sample import Sample


def default_scorers() -> list[Scorer]:
    # 速度不再作为「打分维度」（阈值未定，卡阈值算分不直观）；
    # 改为在分数卡里单列原始时延统计（见 aggregate_scorecard 的 timing）。
    return [
        InstructionScorer(),
        TaskCorrectnessScorer(),
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
        messages = prompt_builder.build_messages(sample)
        generation = backend.generate(messages)
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
                total_ms=generation.total_ms,
                first_token_ms=generation.first_token_ms,
                tokens_per_s=generation.tokens_per_s,
            )
        )
    return aggregate_scorecard(model=backend.model, cases=case_results)
