from slot_extractor.evaluation.scorers.preferences import (
    PreferenceMatcher,
    PreferenceSemanticScorer,
)
from slot_extractor.schemas.results import GenerationResult
from slot_extractor.schemas.sample import Sample


class _FakeEmbedder:
    def __init__(self, scores: dict[tuple[str, str], float]) -> None:
        self.scores = scores

    def similarity(self, left: str, right: str) -> float:
        return self.scores.get((left, right), self.scores.get((right, left), 0.0))


def _sample(preferences: list[str]) -> Sample:
    return Sample(
        id="preference-case",
        layer="final",
        input={},
        expected={"action": "final", "preferences": preferences},
        assertions=[],
        tags=[],
    )


def _result(preferences: list[str]) -> GenerationResult:
    import json

    output = {
        "action": "final",
        "gender": None,
        "start_time": None,
        "duration_minutes": None,
        "preferences": preferences,
        "technician_name": None,
        "technician_status": "not_checked",
        "confirmation": False,
        "info_complete": False,
        "unrelated": False,
        "missing_info": ["start_time", "duration_minutes"],
    }
    return GenerationResult(
        text=json.dumps(output, ensure_ascii=False),
        model="mock",
        prefill_ms=1,
        first_token_ms=1,
        total_ms=1,
    )


def test_preference_matcher_accepts_known_semantic_aliases() -> None:
    matcher = PreferenceMatcher()

    assert matcher.matches("肩颈", "肩膀和脖子") is True
    assert matcher.matches("足部", "脚底按摩") is True
    assert matcher.matches("手法轻柔", "力度轻一点") is True


def test_preference_matcher_rejects_opposite_meaning() -> None:
    matcher = PreferenceMatcher()

    assert matcher.matches("手法轻柔", "力度大一些") is False
    assert matcher.matches("喜欢精油", "不要精油") is False


def test_preference_matcher_uses_vector_fallback_for_unlisted_wording() -> None:
    matcher = PreferenceMatcher(
        embedder=_FakeEmbedder({("重点腰背放松", "腰背放松"): 0.88})
    )

    assert matcher.matches("重点腰背放松", "腰背放松") is True


def test_preference_matcher_accepts_embedding_equivalence_and_rejects_negation() -> None:
    matcher = PreferenceMatcher(
        embedder=_FakeEmbedder(
            {
                ("肩颈", "肩颈按摩"): 0.84,
                ("足部", "足部按摩"): 0.79,
                ("肩颈", "不要按摩肩颈"): 0.90,
            }
        )
    )

    assert matcher.matches("肩颈", "肩颈按摩") is True
    assert matcher.matches("足部", "足部按摩") is True
    assert matcher.matches("肩颈", "不要按摩肩颈") is False


def test_preference_scorer_uses_one_to_one_f1() -> None:
    scorer = PreferenceSemanticScorer()

    full = scorer.score(_sample(["肩颈", "手法轻柔"]), _result(["颈肩放松", "力度轻一点"]))
    missing = scorer.score(_sample(["肩颈", "手法轻柔"]), _result(["颈肩放松"]))
    extra = scorer.score(_sample(["肩颈"]), _result(["颈肩放松", "足部"]))

    assert full.score == 1.0
    assert full.passed is True
    assert missing.score == 2 / 3
    assert extra.score == 2 / 3


def test_preference_scorer_empty_only_matches_empty() -> None:
    scorer = PreferenceSemanticScorer()

    assert scorer.score(_sample([]), _result([])).score == 1.0
    assert scorer.score(_sample([]), _result(["肩颈"])).score == 0.0
