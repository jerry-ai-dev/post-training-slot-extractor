from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from slot_extractor.schemas.sample import validate_context, validate_history

CATEGORY_TAGS = {"追问", "工具调用", "最终 JSON", "确认", "无关"}
DPO_TARGETS_BY_CATEGORY = {
    "追问": frozenset({"P7"}),
    "工具调用": frozenset({"P6", "P2P3"}),
    "最终 JSON": frozenset({"P4", "P2P3"}),
    "确认": frozenset({"P5"}),
    "无关": frozenset({"P5", "P6"}),
}
RAW_FIELDS = {
    "id",
    "output_kind",
    "conversation_kind",
    "tags",
    "input",
    "expected",
    "dpo_targets",
}


@dataclass(frozen=True)
class RawSample:
    id: str
    output_kind: Literal["final", "tool_call"]
    conversation_kind: Literal["single_turn", "multi_turn"]
    tags: tuple[str, ...]
    input: dict[str, Any]
    expected: dict[str, Any]
    dpo_targets: tuple[str, ...]

    @property
    def category(self) -> str:
        return next(tag for tag in self.tags if tag in CATEGORY_TAGS)


def raw_sample_from_record(record: dict[str, Any]) -> RawSample:
    if set(record) != RAW_FIELDS:
        raise ValueError("raw sample top-level fields must equal the seven-field contract")
    sample_id = record["id"]
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("raw sample id must be non-empty text")
    output_kind = record["output_kind"]
    if output_kind not in {"final", "tool_call"}:
        raise ValueError(f"{sample_id} has unsupported output_kind")
    conversation_kind = record["conversation_kind"]
    if conversation_kind not in {"single_turn", "multi_turn"}:
        raise ValueError(f"{sample_id} has unsupported conversation_kind")
    tags = record["tags"]
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise ValueError(f"{sample_id} tags must be a string list")
    categories = CATEGORY_TAGS.intersection(tags)
    if len(categories) != 1:
        raise ValueError(f"{sample_id} must have exactly one category tag")
    input_obj = record["input"]
    expected = record["expected"]
    if not isinstance(input_obj, dict) or not isinstance(expected, dict):
        raise ValueError(f"{sample_id} input and expected must be objects")
    if expected.get("action") != output_kind:
        raise ValueError(f"{sample_id} expected.action must equal output_kind")
    targets = record["dpo_targets"]
    if not isinstance(targets, list) or not all(isinstance(item, str) for item in targets):
        raise ValueError(f"{sample_id} dpo_targets must be a string list")
    if len(targets) != len(set(targets)):
        raise ValueError(f"{sample_id} dpo_targets must not contain duplicates")
    category = next(iter(categories))
    if not set(targets).issubset(DPO_TARGETS_BY_CATEGORY[category]):
        raise ValueError(f"{sample_id} dpo_targets are invalid for {category}")
    validate_context(sample_id, input_obj)
    validate_history(sample_id, input_obj)
    return RawSample(
        id=sample_id,
        output_kind=cast(Literal["final", "tool_call"], output_kind),
        conversation_kind=cast(Literal["single_turn", "multi_turn"], conversation_kind),
        tags=tuple(tags),
        input=input_obj,
        expected=expected,
        dpo_targets=tuple(targets),
    )
