import pytest

from slot_extractor.evaluation.reply_semantics import (
    has_speech_act,
    max_reference_similarity,
    normalize_reply,
    semantic_reply_score,
)
from slot_extractor.schemas.sample import ReplyExpectations


def test_normalize_reply_maps_duration_synonyms() -> None:
    assert normalize_reply("可以按摩一小时") == normalize_reply("可以按摩60分钟")
    assert normalize_reply("安排半小时") == normalize_reply("安排30分钟")


def test_normalize_reply_maps_common_chinese_time_numerals() -> None:
    assert normalize_reply("明天下午两点") == normalize_reply("明天下午2点")


@pytest.mark.parametrize(
    "text",
    [
        "您确认一下可以吗？",
        "这个安排您看可以吗？",
        "需要为您安排吗？",
    ],
)
def test_detects_confirmation_request_paraphrases(text: str) -> None:
    assert has_speech_act(text, "request_confirmation")


def test_detects_premature_booking_success_claim() -> None:
    assert has_speech_act("已经帮您预约成功了", "claim_booking_success")


def test_detects_pause_acknowledgement() -> None:
    assert has_speech_act("好的，暂时不给您预约，需要时再告诉我。", "acknowledge_pause")


@pytest.mark.parametrize(
    ("text", "act"),
    [
        ("按摩时长您方便说一下吗？", "ask_for_duration"),
        ("王芳明天下午两点可以服务，您确认吗？", "inform_technician_available"),
        ("抱歉，未能找到陈静技师。", "inform_technician_not_found"),
        ("目前没有找到会泰式的女技师。", "inform_no_match"),
        ("已为您暂停预约，可随时重新安排。", "acknowledge_pause"),
    ],
)
def test_detects_dataset_reply_paraphrases(text: str, act: str) -> None:
    assert has_speech_act(text, act)


@pytest.mark.parametrize(
    ("text", "act"),
    [
        ("周末下午的具体时间您方便说一下吗？", "ask_for_start_time"),
        ("已找到王芳技师，明天下午两点进行按摩。", "inform_technician_available"),
        ("已授权创建预约", "acknowledge_booking_authorization"),
    ],
)
def test_detects_additional_valid_paraphrases(text: str, act: str) -> None:
    assert has_speech_act(text, act)


def test_unavailable_wording_is_not_also_available() -> None:
    text = "李明技师在明天下午三点不可用。"

    assert has_speech_act(text, "inform_technician_unavailable")
    assert not has_speech_act(text, "inform_technician_available")


def test_common_natural_paraphrases_match_required_acts() -> None:
    assert has_speech_act("请问您想预约什么具体时间？", "ask_for_start_time")
    assert has_speech_act("王芳技师明天下午可以提供60分钟服务。", "inform_technician_available")
    assert has_speech_act("李明技师该时段无法提供服务。", "inform_technician_unavailable")
    assert has_speech_act("未找到名为陈静的技师。", "inform_technician_not_found")
    assert has_speech_act("好的，已确认按该方案提交预约。", "acknowledge_booking_authorization")
    assert has_speech_act("请上层创建预约。", "acknowledge_booking_authorization")
    assert has_speech_act(
        "李明技师可在6月10日上午10点提供60分钟服务。",
        "inform_technician_available",
    )


def test_reference_similarity_accepts_paraphrase_better_than_unrelated_text() -> None:
    references = (
        "请问您想什么时候过来呢？",
        "您希望预约什么时间？",
    )

    paraphrase = max_reference_similarity("您打算几点过来？", references)
    unrelated = max_reference_similarity("今天天气不错。", references)

    assert paraphrase > unrelated


def test_semantic_reply_score_requires_all_acts_and_no_forbidden_act() -> None:
    expectations = ReplyExpectations(
        required_acts=("inform_technician_available", "request_confirmation"),
        forbidden_acts=("claim_booking_success",),
        required_fields=("technician_name",),
        references=("王芳技师有空，您确认吗？",),
    )

    score, passed, detail = semantic_reply_score("王芳技师有空，这个安排您看可以吗？", expectations)

    assert score >= 0.7
    assert passed is True
    assert "required=2/2" in detail


def test_semantic_reply_score_rejects_forbidden_claim() -> None:
    expectations = ReplyExpectations(
        required_acts=("inform_technician_available",),
        forbidden_acts=("claim_booking_success",),
        required_fields=(),
        references=("王芳技师有空。",),
    )

    _, passed, detail = semantic_reply_score("王芳技师有空，已经预约成功。", expectations)

    assert passed is False
    assert "forbidden=claim_booking_success" in detail


def test_semantic_reply_score_accepts_full_act_coverage_with_low_text_overlap() -> None:
    expectations = ReplyExpectations(
        required_acts=("acknowledge_pause",),
        forbidden_acts=("claim_booking_success",),
        required_fields=(),
        references=("好的，这次先不预约，需要时再联系我。",),
    )

    _, passed, detail = semantic_reply_score("已暂停预约，您可随时重新安排。", expectations)

    assert passed is True
    assert "required=1/1" in detail
