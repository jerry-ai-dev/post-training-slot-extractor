from __future__ import annotations

from slot_extractor.evaluation.reply_semantics import semantic_reply_score
from slot_extractor.schemas.output import OutputValidationError, parse_model_json
from slot_extractor.schemas.results import DimensionScore, GenerationResult
from slot_extractor.schemas.sample import Sample


class ReplySemanticScorer:
    dimension = "reply_semantic"

    def applies_to(self, sample: Sample) -> bool:
        return sample.expected.get("action") == "final" and sample.reply_expectations is not None

    def score(self, sample: Sample, result: GenerationResult) -> DimensionScore:
        if sample.reply_expectations is None:
            return DimensionScore(self.dimension, None, None, "no reply expectations")
        try:
            output = parse_model_json(result.text)
        except OutputValidationError as exc:
            return DimensionScore(self.dimension, 0.0, False, str(exc))
        reply = output.get("reply")
        if reply is None and sample.expected.get("reply_type") == "handoff":
            return DimensionScore(self.dimension, 1.0, True, "handoff has no local reply")
        if not isinstance(reply, str) or not reply.strip():
            return DimensionScore(self.dimension, 0.0, False, "reply is missing")
        score, passed, detail = semantic_reply_score(reply, sample.reply_expectations)
        return DimensionScore(self.dimension, score, passed, detail)
