from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Any, Protocol

from slot_extractor.schemas.output import OutputValidationError, parse_model_json
from slot_extractor.schemas.results import DimensionScore, GenerationResult
from slot_extractor.schemas.sample import Sample

SIMILARITY_THRESHOLD = 0.70
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
_PUNCTUATION = re.compile(r"[\s，。！？、,.!?;；:：·\-_/]+")
_NEGATIONS = ("不要", "不想", "不喜欢", "避免", "别用", "不能", "不需要")
_ALIASES = {
    "肩颈": {"肩颈", "颈肩", "肩膀和脖子", "脖子和肩膀", "肩颈放松", "颈肩放松"},
    "足部": {"足部", "足疗", "脚底按摩", "按脚", "脚部按摩", "足底"},
    "精油": {"精油", "精油按摩", "精油spa", "芳香精油", "喜欢精油"},
    "泰式": {"泰式", "泰式按摩"},
    "手法轻柔": {"手法轻柔", "力度轻一点", "轻柔", "轻一点", "不要太重"},
    "力度较重": {"力度大一些", "重一点", "手法重", "力度重"},
}


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower().strip()
    return _PUNCTUATION.sub("", normalized)


_NORMALIZED_ALIASES = {
    _normalize(alias): canonical for canonical, aliases in _ALIASES.items() for alias in aliases
}


def _polarity(text: str) -> bool:
    normalized = _normalize(text)
    return not any(negation in normalized for negation in _NEGATIONS)


class TextEmbedder(Protocol):
    def similarity(self, left: str, right: str) -> float: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")
        self._cache: dict[str, Any] = {}

    def _encode(self, text: str):
        normalized = _normalize(text)
        if normalized not in self._cache:
            self._cache[normalized] = self._model.encode(
                normalized, normalize_embeddings=True
            )
        return self._cache[normalized]

    def similarity(self, left: str, right: str) -> float:
        left_vector = self._encode(left)
        right_vector = self._encode(right)
        return float(left_vector @ right_vector)


@lru_cache(maxsize=1)
def _default_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


class PreferenceMatcher:
    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        embedder: TextEmbedder | None = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self._embedder = embedder

    def similarity(self, expected: str, actual: str) -> float:
        expected_normalized = _normalize(expected)
        actual_normalized = _normalize(actual)
        if expected_normalized == actual_normalized:
            return 1.0
        expected_alias = _NORMALIZED_ALIASES.get(expected_normalized)
        actual_alias = _NORMALIZED_ALIASES.get(actual_normalized)
        if expected_alias is not None and expected_alias == actual_alias:
            return 1.0
        if expected_alias is not None and actual_alias is not None:
            return 0.0
        if _polarity(expected) != _polarity(actual):
            return 0.0
        embedder = self._embedder or _default_embedder()
        return embedder.similarity(expected, actual)

    def matches(self, expected: str, actual: str) -> bool:
        return self.similarity(expected, actual) >= self.similarity_threshold


class PreferenceSemanticScorer:
    dimension = "preferences_semantic"

    def __init__(self, matcher: PreferenceMatcher | None = None) -> None:
        self.matcher = matcher or PreferenceMatcher()

    def applies_to(self, sample: Sample) -> bool:
        return "preferences" in sample.expected

    def score(self, sample: Sample, result: GenerationResult) -> DimensionScore:
        expected = sample.expected.get("preferences")
        if not isinstance(expected, list) or not all(isinstance(item, str) for item in expected):
            return DimensionScore(self.dimension, None, None, "n/a: invalid expected preferences")
        try:
            output = parse_model_json(result.text)
        except OutputValidationError as exc:
            return DimensionScore(self.dimension, 0.0, False, str(exc))
        actual = output.get("preferences")
        if not isinstance(actual, list) or not all(isinstance(item, str) for item in actual):
            return DimensionScore(self.dimension, 0.0, False, "preferences must be string array")
        if not expected and not actual:
            return DimensionScore(self.dimension, 1.0, True, "empty preferences match")
        if not expected or not actual:
            return DimensionScore(self.dimension, 0.0, False, "one side has no preferences")

        candidates = sorted(
            (
                (
                    self.matcher.similarity(expected_value, actual_value),
                    expected_index,
                    actual_index,
                )
                for expected_index, expected_value in enumerate(expected)
                for actual_index, actual_value in enumerate(actual)
            ),
            reverse=True,
        )
        matched_expected: set[int] = set()
        matched_actual: set[int] = set()
        for similarity, expected_index, actual_index in candidates:
            if similarity < self.matcher.similarity_threshold:
                break
            if expected_index in matched_expected or actual_index in matched_actual:
                continue
            matched_expected.add(expected_index)
            matched_actual.add(actual_index)

        matches = len(matched_expected)
        precision = matches / len(actual)
        recall = matches / len(expected)
        score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return DimensionScore(
            self.dimension,
            score,
            score == 1.0,
            f"matches={matches} precision={precision:.3f} recall={recall:.3f}",
        )
