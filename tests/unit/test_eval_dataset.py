import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from slot_extractor.evaluation.assertions import evaluate_assertion
from slot_extractor.schemas.dataset_contract import (
    load_dataset_contract,
    validate_dataset_against_contract,
)
from slot_extractor.schemas.sample import load_samples

DATASET = Path("data/eval/test.jsonl")
CONTRACT = Path("data/eval/dataset_contract.json")


def _samples():
    return load_samples(DATASET)


def _tool_exchanges(sample):
    calls = {}
    exchanges = []
    for turn in sample.input["history"]:
        if turn["role"] == "assistant" and "tool_calls" in turn:
            for call in turn["tool_calls"]:
                calls[call["id"]] = {
                    "tool_name": call["function"]["name"],
                    "arguments": json.loads(call["function"]["arguments"]),
                }
        elif turn["role"] == "tool":
            exchanges.append(
                {
                    **calls[turn["tool_call_id"]],
                    "result": json.loads(turn["content"]),
                }
            )
    return exchanges


def test_dataset_keeps_all_51_cases() -> None:
    assert len(_samples()) == 51


def test_dataset_checksum_matches() -> None:
    expected = Path("data/eval/test.sha256").read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    assert actual == expected


def test_every_assertion_is_parseable_against_expected() -> None:
    for sample in _samples():
        for expression in sample.assertions:
            result = evaluate_assertion(expression, dict(sample.expected), sample)
            assert result.passed, f"{sample.id}: {expression} -> {result.detail}"


def test_dataset_satisfies_contract() -> None:
    contract = load_dataset_contract(CONTRACT)
    assert contract["version"] == "2.1"
    validate_dataset_against_contract(_samples(), contract)


def test_every_case_uses_the_dual_track_runtime_input() -> None:
    expected_keys = {
        "history",
        "current_state",
        "user_input",
        "current_time",
        "available_tools",
    }
    for sample in _samples():
        assert set(sample.input) == expected_keys, sample.id
        assert sample.input["available_tools"] == ["find_technicians"], sample.id


def test_history_contains_valid_natural_and_tool_messages() -> None:
    for sample in _samples():
        for turn in sample.input["history"]:
            assert turn["role"] in {"user", "assistant", "tool"}, sample.id
            if turn["role"] == "assistant" and "tool_calls" not in turn:
                try:
                    value = json.loads(turn["content"])
                except json.JSONDecodeError:
                    continue
                assert not isinstance(value, (dict, list)), sample.id


def test_every_final_has_reply_contract_and_every_tool_call_does_not() -> None:
    for sample in _samples():
        if sample.expected["action"] == "final":
            assert "reply_type" in sample.expected, sample.id
            assert "reply" in sample.expected, sample.id
            assert sample.reply_expectations is not None, sample.id
        else:
            assert "reply_type" not in sample.expected, sample.id
            assert "reply" not in sample.expected, sample.id
            assert sample.reply_expectations is None, sample.id


def test_all_reply_types_and_tool_results_are_covered() -> None:
    reply_types = {
        sample.expected["reply_type"]
        for sample in _samples()
        if sample.expected["action"] == "final"
    }
    assert {
        "handoff",
        "ask_start_time",
        "ask_duration",
        "ask_start_time_and_duration",
        "confirm_available",
        "inform_unavailable",
        "inform_not_found",
        "inform_no_match",
        "booking_authorized",
        "acknowledge_result",
        "appointment_paused",
    } == reply_types

    tool_states = {
        (exchange["result"]["mode"], exchange["result"]["status"])
        for sample in _samples()
        for exchange in _tool_exchanges(sample)
    }
    assert {
        ("specific", "available"),
        ("specific", "unavailable"),
        ("specific", "not_found"),
        ("search", "matched"),
        ("search", "no_match"),
    } <= tool_states


def test_every_tool_call_has_successor_with_exact_tool_exchange() -> None:
    chains = defaultdict(dict)
    for sample in _samples():
        if sample.chain_id is not None:
            chains[sample.chain_id][sample.step] = sample

    for sample in _samples():
        if sample.expected["action"] != "tool_call":
            continue
        assert sample.chain_id is not None and sample.step is not None, sample.id
        successor = chains[sample.chain_id].get(sample.step + 1)
        assert successor is not None, sample.id
        exchange = _tool_exchanges(successor)[-1]
        assert exchange["tool_name"] == sample.expected["tool_name"], sample.id
        assert exchange["arguments"] == sample.expected["arguments"], sample.id


def test_repeated_queries_keep_every_tool_exchange() -> None:
    expected_counts = {
        "plan-modify-01-result": 2,
        "plan-modify-time-pref-01-requery-result": 2,
        "search-no-match-01-retry-result": 2,
        "specific-not-found-01-reselect-result": 2,
        "specific-unavailable-01-reselect-result": 2,
    }
    samples = {sample.id: sample for sample in _samples()}

    for sample_id, expected_count in expected_counts.items():
        assert len(_tool_exchanges(samples[sample_id])) == expected_count


def test_checked_current_state_has_tool_history_evidence() -> None:
    for sample in _samples():
        state = sample.input["current_state"]
        if isinstance(state, dict) and state["technician_status"] != "not_checked":
            assert _tool_exchanges(sample), sample.id


def test_tool_result_turn_state_matches_latest_call_arguments() -> None:
    fields = ("gender", "start_time", "duration_minutes", "preferences", "technician_name")
    for sample in _samples():
        if not sample.input["history"] or sample.input["history"][-1]["role"] != "tool":
            continue
        latest = _tool_exchanges(sample)[-1]
        state = sample.input["current_state"]
        assert state is not None, sample.id
        for field in fields:
            assert state[field] == latest["arguments"][field], sample.id
        assert state["technician_status"] == "not_checked", sample.id
        assert state["info_complete"] is True, sample.id
        assert state["missing_info"] == [], sample.id


def test_checked_plan_changes_trigger_a_new_tool_call() -> None:
    cases = [
        sample
        for sample in _samples()
        if sample.expected["action"] == "tool_call"
        and isinstance(sample.input["current_state"], dict)
        and sample.input["current_state"]["technician_status"] != "not_checked"
    ]
    assert {sample.id for sample in cases} == {
        "plan-modify-01-call",
        "plan-modify-time-pref-01-requery",
        "search-no-match-01-retry",
        "specific-not-found-01-reselect",
        "specific-unavailable-01-reselect",
    }
    for sample in cases:
        previous = _tool_exchanges(sample)[-1]["arguments"]
        assert previous != sample.expected["arguments"], sample.id


def test_time_and_preference_change_uses_second_query_result() -> None:
    samples = {sample.id: sample for sample in _samples()}
    requery = samples["plan-modify-time-pref-01-requery"]
    result = samples["plan-modify-time-pref-01-requery-result"]
    exchanges = _tool_exchanges(result)

    assert exchanges[0]["arguments"]["start_time"] == "2026-06-09 14:00"
    assert exchanges[0]["arguments"]["preferences"] == ["肩颈"]
    assert requery.expected["arguments"]["start_time"] == "2026-06-10 16:00"
    assert requery.expected["arguments"]["preferences"] == ["精油"]
    assert exchanges[-1]["arguments"] == requery.expected["arguments"]
    assert result.expected["start_time"] == "2026-06-10 16:00"
    assert result.expected["preferences"] == ["精油"]
    assert result.expected["technician_name"] == "王芳"
    assert result.expected["technician_status"] == "available"


def test_confirmation_uses_matching_pending_current_state() -> None:
    fields = (
        "gender",
        "start_time",
        "duration_minutes",
        "preferences",
        "technician_name",
        "technician_status",
    )
    confirmation_cases = [
        sample for sample in _samples() if sample.expected.get("confirmation") is True
    ]
    assert confirmation_cases
    for sample in confirmation_cases:
        state = sample.input["current_state"]
        assert state is not None, sample.id
        for field in fields:
            assert state[field] == sample.expected[field], sample.id


def test_tool_calls_require_known_time_and_duration() -> None:
    for sample in _samples():
        if sample.expected["action"] == "tool_call":
            arguments = sample.expected["arguments"]
            assert arguments["start_time"] is not None, sample.id
            assert isinstance(arguments["duration_minutes"], int), sample.id


def test_reply_type_distribution_is_not_single_template_dominated() -> None:
    counts = Counter(
        sample.expected.get("reply_type")
        for sample in _samples()
        if sample.expected["action"] == "final"
    )
    assert counts["confirm_available"] < sum(counts.values()) / 2


def test_vague_period_does_not_invent_a_specific_hour() -> None:
    sample = next(item for item in _samples() if item.id == "ask-0001")

    assert sample.expected["start_time"] is None
    assert sample.expected["missing_info"] == ["start_time", "duration_minutes"]
    assert sample.expected["reply_type"] == "ask_start_time_and_duration"
